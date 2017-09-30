package algorithm;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class MyID3BackupJelek 
	extends Classifier 
	implements TechnicalInformationHandler, Sourcable {
	
	/** The node's successors. */ 
	private MyID3BackupJelek[] node_Successors;

	/** Attribute used for splitting. */
	private Attribute node_Attribute;
	
	/** Class value if node is leaf. */
	private double node_ClassValue;
	
	/** Class distribution if node is leaf. */
	private double[] node_Distribution;


	@Override
	public String toSource(String arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;
		result = new TechnicalInformation(Type.ARTICLE);
	    result.setValue(Field.AUTHOR, "Arizho, Faiq, Wiega");
	    result.setValue(Field.YEAR, "2017");
	    result.setValue(Field.TITLE, "Custom ID3");
	    result.setValue(Field.JOURNAL, "Machine Learning");
	    return result;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
	    getCapabilities().testWithFail(data);

	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();
	    
	    List<Attribute> list_attributes = new ArrayList<Attribute>();
	    makeTree(data, list_attributes);
		
	}
	
	  /**
	   * Returns default capabilities of the classifier.
	   *
	   * @return      the capabilities of this classifier
	   */
	  public Capabilities getCapabilities() {
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.MISSING_CLASS_VALUES);

	    // instances
	    result.setMinimumNumberInstances(0);
	    
	    return result;
	  }
	
	/**
	* Splits a dataset according to the values of a nominal attribute.
	*
	* @param data the data which is to be split
	* @param att the attribute to be used for splitting
	* @return the sets of instances produced by the split
	*/
	private Instances[] splitData(Instances data, Attribute att) {
	
	  Instances[] split_data = new Instances[att.numValues()];
	  for (int j = 0; j < att.numValues(); j++) {
		  split_data[j] = new Instances(data, data.numInstances());
	  }
	  Enumeration instEnum = data.enumerateInstances();
	  while (instEnum.hasMoreElements()) {
	    Instance inst = (Instance) instEnum.nextElement();
	    split_data[(int) inst.value(att)].add(inst);
	  }
	  for (int i = 0; i < split_data.length; i++) {
		  split_data[i].compactify();
	  }
	  return split_data;
	}

	
	private void makeTree(Instances data, List<Attribute> list_attributes) throws Exception {
		
		//If all Examples are uniform, Return the single-node tree Root, with same label
		System.out.println("Make Tree");
		if (isUniformInstances(data)) {
			node_ClassValue = data.instance(0).classValue(); 
			node_Attribute = null;
	    	node_Distribution = new double[data.numClasses()];
	        Enumeration instEnum = data.enumerateInstances();
	        while (instEnum.hasMoreElements()) {
	          Instance inst = (Instance) instEnum.nextElement();
	          node_Distribution[(int) inst.classValue()]++;
	        }
	        Utils.normalize(node_Distribution);

			return;
		}
		// If Attributes is empty, Return the single-node tree Root,
		// with label = most common value of ClassAtribute in Examples
		else if (data.numAttributes() == 0) {
	    	// Make this node as Leaf
//			node_ClassValue = getMostCommonValue(data); 
			node_ClassValue = Instance.missingValue();
			node_Attribute = null;
	    	node_Distribution = new double[data.numClasses()];
	    	Enumeration instEnum = data.enumerateInstances();
	        while (instEnum.hasMoreElements()) {
	          Instance inst = (Instance) instEnum.nextElement();
	          node_Distribution[(int) inst.classValue()]++;
	        }
	        Utils.normalize(node_Distribution);

			return;
	    }
	    else {
	    	
	    	// Let decision attribute of node =  the attribute that best classifies Instances
//	    	node_Attribute = getBestAttribute(data);
	    	// Compute attribute with maximum information gain.
	    	double[] infoGains = new double[list_attributes.size()];
	        
	        for (int i=0; i < list_attributes.size(); i++){
	        	Attribute att = (Attribute) list_attributes.get(i);
	        	infoGains[i] = computeInfoGain(data, att);
	        }
	        
	        node_Attribute = data.attribute(Utils.maxIndex(infoGains));
	        System.out.println("nodeAttribute : " + node_Attribute);
	    	// For each possible value of node_Attribute
	    	Instances[] subset_data = splitData(data,node_Attribute);
	    	node_Successors = new MyID3BackupJelek[subset_data.length];
	    	for (int i=0; i < subset_data.length; i++) {
    			node_Successors[i] = new MyID3BackupJelek();
	    		if (subset_data[i].numInstances() == 0){
	    			System.out.println("nodsuc at "+i +" leafnode");
	    			// Add a leaf node with label = most common value of ClassAtribute in Examples
//	    			node_Successors[i].node_ClassValue = getMostCommonValue(subset_data[i]); 
	    			node_Successors[i].node_ClassValue = Instance.missingValue();
	    			node_Successors[i].node_Attribute = null;
	    			node_Successors[i].node_Distribution = new double[subset_data[i].numClasses()];
	    			Enumeration instEnum = subset_data[i].enumerateInstances();
	    	        while (instEnum.hasMoreElements()) {
	    	          Instance inst = (Instance) instEnum.nextElement();
	    	          node_Successors[i].node_Distribution[(int) inst.classValue()]++;
	    	        }
	    	        Utils.normalize(node_Successors[i].node_Distribution);
	    			return;
	    		}
	    		else {
	    			System.out.println("nodsuc at "+i+" createbranch");
	    			// Add a subtree ID3(subset_data[i])
	    			list_attributes.remove(node_Attribute);
	    			node_Successors[i].makeTree(subset_data[i], list_attributes);
	    		}
	    	}
	    	
	    }
		
	}

	private String getMostCommonValue(Instances data) {
		int[] counter = new int[data.numClasses()];
		String[] target = new String[data.numClasses()];
		Enumeration en_val = data.classAttribute().enumerateValues();
		int k = 0;
		while (en_val.hasMoreElements()){
			target[k] = (String) en_val.nextElement();
			k++;
		}
		
		for (int i = 0; i < data.numInstances(); i++){
			for (int j = 0; j < target.length; j++){
				if (data.instance(i).stringValue(i) == target[j]){
					counter[j]++;
					break;
				}
			}
		}
		
		int max_idx = 0;
		
		for (int j = 0; j < counter.length; j++){
			if (counter[j] > counter[max_idx]){
				max_idx = j;
			}
		}
		
		return target[max_idx];
	}

	private boolean isUniformInstances(Instances data) {
		boolean check = true;
		if (data.numInstances() > 0){
			double comparator = data.instance(0).classValue();
			for (int i=0 ; i < data.numInstances(); i++){
				if (data.instance(i).classValue() != comparator) {
					check = false;
					break;
				}
			}
			return check;
		}
		else {
			return false;
		}
	}

	private double[] getInfoGains(Instances data) throws Exception {
		double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
          Attribute att = (Attribute) attEnum.nextElement();
          infoGains[att.index()] = computeInfoGain(data, att);
        }
        
        return infoGains;
	}

	 /**
	  * Computes information gain for an attribute.
	  *
	  * @param data the data for which info gain is to be computed
	  * @param att the attribute
	  * @return the information gain for the given attribute and data
	  * @throws Exception if computation fails
	  */
	 private double computeInfoGain(Instances data, Attribute att) 
	   throws Exception {

	   double infoGain = computeEntropy(data);
	   Instances[] split_data = splitData(data, att);
	   for (int j = 0; j < att.numValues(); j++) {
	     if (split_data[j].numInstances() > 0) {
	       infoGain -= ((double) split_data[j].numInstances() /
	                    (double) data.numInstances()) *
	         computeEntropy(split_data[j]);
	     }
	   }
	   return infoGain;
	 }
	
	  /**
	   * Computes the entropy of a dataset.
	   * 
	   * @param data the data for which entropy is to be computed
	   * @return the entropy of the data's class distribution
	   * @throws Exception if computation fails
	   */
	  private double computeEntropy(Instances data) throws Exception {

	    double [] classCounts = new double[data.numClasses()];
	    Enumeration instEnum = data.enumerateInstances();
	    while (instEnum.hasMoreElements()) {
	      Instance inst = (Instance) instEnum.nextElement();
	      classCounts[(int) inst.classValue()]++;
	    }
	    double entropy = 0;
	    for (int j = 0; j < data.numClasses(); j++) {
	      if (classCounts[j] > 0) {
	        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
	      }
	    }
	    entropy /= (double) data.numInstances();
	    return entropy + Utils.log2(data.numInstances());
	  }

	  /**
	   * Classifies a given test instance using the decision tree.
	   *
	   * @param instance the instance to be classified
	   * @return the classification
	   * @throws NoSupportForMissingValuesException if instance has missing values
	   */
	  public double classifyInstance(Instance instance) 
	    throws NoSupportForMissingValuesException {

	    if (instance.hasMissingValue()) {
	      throw new NoSupportForMissingValuesException("Id3: no missing values, "
	                                                   + "please.");
	    }
	    if (node_Attribute == null) {
	      return node_ClassValue;
	    } else {
	      return node_Successors[(int) instance.value(node_Attribute)].
	        classifyInstance(instance);
	    }
	  }
	  
}
