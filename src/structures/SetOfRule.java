package structures;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class SetOfRule {
	private List<Rule> set_of_rule;
	private double accuracy;
	private Instances train_data;
	
	public SetOfRule(Instances data){
		train_data = data;
		set_of_rule = new ArrayList<Rule>();
		accuracy = 0;
	}
	
	public double getAccuracy() {
		return accuracy;
	}
	
	public void addPrecondition(Rule rule){
		set_of_rule.add(rule);
	}
	
	public void delPrecondition(Rule rule){
		set_of_rule.remove(rule);
	}
	
	private void countAccuracy(){
		accuracy = 0;
	}
	
	public double classify(Instance instance){
		// NOT IMPLEMENTED		
		return 0;
	}
}
