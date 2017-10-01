package algorithm;

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
	private MyID3 decision_tree;

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
	public void buildClassifier(Instances arg0) throws Exception {
		
		
	}
	
	@Override
	public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
		return decision_tree.classifyInstance(instance);
		
	}

}
