"""
Unit tests for PSL reasoning.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rules.psl import PSLRule, PSLReasoner, create_default_rules


class TestPSLRule:
    """Test PSL rule functionality."""
    
    def test_rule_creation(self):
        """Test creating a PSL rule."""
        rule = PSLRule(
            name="test_rule",
            antecedent="A(x)",
            consequent="B(x)",
            weight=0.8,
            rule_type="default"
        )
        
        assert rule.name == "test_rule"
        assert rule.antecedent == "A(x)"
        assert rule.consequent == "B(x)"
        assert rule.weight == 0.8
        assert rule.rule_type == "default"
    
    def test_rule_string_representation(self):
        """Test rule string representation."""
        rule = PSLRule("test", "A(x)", "B(x)", 0.8)
        expected = "A(x) -> B(x) [w=0.8]"
        assert str(rule) == expected
    
    def test_invalid_weight(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError):
            PSLRule("test", "A(x)", "B(x)", -0.5)


class TestPSLReasoner:
    """Test PSL reasoner functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.rules = [
            PSLRule("rule1", "A(x)", "B(x)", 0.8),
            PSLRule("rule2", "B(x)", "C(x)", 0.9)
        ]
        self.reasoner = PSLReasoner(self.rules, t_norm="lukasiewicz")
    
    def test_reasoner_creation(self):
        """Test creating a PSL reasoner."""
        assert len(self.reasoner.rules) == 2
        assert self.reasoner.t_norm == "lukasiewicz"
        assert self.reasoner.rule_weights.shape[0] == 2
    
    def test_lukasiewicz_satisfaction(self):
        """Test Łukasiewicz t-norm satisfaction."""
        antecedent = torch.tensor([0.8, 0.3, 1.0])
        consequent = torch.tensor([0.9, 0.1, 0.5])
        
        satisfaction = self.reasoner._lukasiewicz_satisfaction(antecedent, consequent)
        expected = torch.tensor([1.0, 0.8, 0.5])  # min(1, 1 - A + B)
        
        torch.testing.assert_close(satisfaction, expected)
    
    def test_godel_satisfaction(self):
        """Test Gödel t-norm satisfaction."""
        antecedent = torch.tensor([0.8, 0.3, 1.0])
        consequent = torch.tensor([0.9, 0.1, 0.5])
        
        satisfaction = self.reasoner._godel_satisfaction(antecedent, consequent)
        expected = torch.tensor([0.9, 0.1, 0.5])  # 1 if A <= B, B otherwise
        
        torch.testing.assert_close(satisfaction, expected)
    
    def test_forward_pass(self):
        """Test forward pass through reasoner."""
        facts = {
            "A(x)": torch.tensor([0.8, 0.5]),
            "B(x)": torch.tensor([0.7, 0.6]),
            "C(x)": torch.tensor([0.9, 0.4])
        }
        
        rule_instances = [
            ("rule1", "A(x)", "B(x)"),
            ("rule2", "B(x)", "C(x)")
        ]
        
        outputs = self.reasoner.forward(facts, rule_instances)
        
        assert "symbolic_scores" in outputs
        assert "rule_satisfactions" in outputs
        assert "mean_satisfaction" in outputs
        assert outputs["symbolic_scores"].shape == (2,)
    
    def test_logic_loss(self):
        """Test logic loss computation."""
        facts = {
            "A(x)": torch.tensor([0.8, 0.5]),
            "B(x)": torch.tensor([0.7, 0.6])
        }
        
        rule_instances = [("rule1", "A(x)", "B(x)")]
        
        loss = self.reasoner.compute_logic_loss(facts, rule_instances)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
    
    def test_rule_weight_updates(self):
        """Test updating rule weights."""
        new_weights = {"rule1": 0.9, "rule2": 0.7}
        self.reasoner.update_rule_weights(new_weights)
        
        current_weights = self.reasoner.get_rule_weights()
        assert current_weights["rule1"] == 0.9
        assert current_weights["rule2"] == 0.7


class TestDefaultRules:
    """Test default rule creation."""
    
    def test_create_default_rules(self):
        """Test creating default commonsense rules."""
        rules = create_default_rules()
        
        assert len(rules) > 0
        assert all(isinstance(rule, PSLRule) for rule in rules)
        
        # Check for specific rule types
        rule_names = [rule.name for rule in rules]
        assert "bird_can_fly" in rule_names
        assert "penguin_cannot_fly" in rule_names
        assert "fish_in_water" in rule_names


if __name__ == "__main__":
    pytest.main([__file__]) 