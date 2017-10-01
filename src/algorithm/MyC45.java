package algorithm;

import java.util.List;

import algorithm.C45Support.MyID3withGainRatio;
import structures.Rule;
import structures.SetOfRule;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class MyC45 extends Classifier 
implements TechnicalInformationHandler, Sourcable {
	
	  /** The decision tree */
	private MyID3withGainRatio decision_tree;
	private SetOfRule set_of_rule;

	@Override
	public String toSource(String arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances final_data = preprocess_data(data);
		
		decision_tree = new MyID3withGainRatio();
		decision_tree.buildClassifier(data);
		
		SetOfRule init_set_of_rule = convertTreeIntoRules(decision_tree);
		
		List<SetOfRule> possible_set_of_rule = generateAllPrunedSetofRule(init_set_of_rule);
		
		set_of_rule = selectBestSetofRule(possible_set_of_rule); 
		
	}
	
	private SetOfRule selectBestSetofRule(List<SetOfRule> possible_set_of_rule) {
		// NOT IMPLEMENTED
		return null;
	}

	private List<SetOfRule> generateAllPrunedSetofRule(SetOfRule rules) {
		// NOT IMPLEMENTED
		return null;
	}

	private SetOfRule convertTreeIntoRules(MyID3withGainRatio decision_tree2) {
		// NOT IMPLEMENTED
		return null;
	}

	private Instances preprocess_data(Instances data) {
		// NOT IMPLEMENTED
		// Implement ini wiega
		return null;
	}

	@Override
	public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
		return set_of_rule.classify(instance);
		
	}

}
