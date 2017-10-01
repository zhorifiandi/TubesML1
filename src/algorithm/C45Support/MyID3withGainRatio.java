package algorithm.C45Support;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Enumeration;

import algorithm.MyID3;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class MyID3withGainRatio extends MyID3 {
	private MyID3withGainRatio[] node_Successors;
		
	private double computeInfoGain(Instances data, Attribute att) 
			   throws Exception {
		
		double infoGain = computeEntropy(data);
		Instances[] split_data = splitData(data, att);
		
		for (int j = 0; j < att.numValues(); j++) {
			if (split_data[j].numInstances() > 0) {
				double missNum = 0;
				for (int k = 0; k < split_data[j].numInstances(); k++) {
					if (split_data[j].instance(k).hasMissingValue()) {
						missNum++;
					}
				}
				double normalNum = split_data[j].numInstances() - missNum;
				double weight = normalNum + (normalNum * missNum / (double) data.numInstances());
				infoGain -= (weight / (double) data.numInstances()) *
						computeEntropy(split_data[j]);
		    }
		}
		
		infoGain = infoGain / computeEntropy(data);
		
		return infoGain;
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
	    if (!inst.hasMissingValue()) {
	    	split_data[(int) inst.value(att)].add(inst);
	    } else {
	    	for (int i = 0; i < split_data.length; i++) {
	    		split_data[i].add(inst);
	    	}
	    }
	  }
	  for (int i = 0; i < split_data.length; i++) {
		  split_data[i].compactify();
	  }
	  return split_data;
	}
	
	private void makeTree(Instances data) throws Exception {
		//If all Examples are uniform, Return the single-node tree Root, with same label 
		if (isUniformInstances(data)) {
			node_ClassValue = data.instance(0).classValue(); 
			node_Attribute = null;
			return;
		}
		// If Attributes is empty, Return the single-node tree Root,
		// with label = most common value of ClassAtribute in Examples
		else if (data.numAttributes() == 0) {
	    	// Make this node as Leaf
			node_ClassValue = getMostCommonValue(data); 
			node_Attribute = null;
			return;
	    }
	    else {
	    	
	    	// Let decision attribute of node =  the attribute that best classifies Instances
	    	// Compute attribute with maximum information gain.
	        double[] infoGains = new double[data.numAttributes()];
	        Enumeration attEnum = data.enumerateAttributes();
	        while (attEnum.hasMoreElements()) {
	          Attribute att = (Attribute) attEnum.nextElement();
	          infoGains[att.index()] = computeInfoGain(data, att);
	        }
	        node_Attribute = data.attribute(Utils.maxIndex(infoGains));
	        
	    	// For each possible value of node_Attribute
	    	Instances[] subset_data = splitData(data,node_Attribute);
	    	node_Successors = new MyID3withGainRatio[subset_data.length];
	    	for (int i=0; i < subset_data.length; i++) {
    			node_Successors[i] = new MyID3withGainRatio();
	    		if (subset_data[i].numInstances() == 0){
	    			// Add a leaf node with label = most common value of ClassAtribute in Examples
	    			node_Successors[i].node_ClassValue = getMostCommonValue(subset_data[i]); 
	    			node_Successors[i].node_Attribute = null;
	    			return;
	    		}
	    		else {
	    			// Add a subtree ID3(subset_data[i])
	    			node_Successors[i].makeTree(subset_data[i]);
	    		}
	    	}
	    	
	    }
		
	}
	
	public static void main(String[] args) throws Exception {
		BufferedReader breader = new BufferedReader(new FileReader("arff//weather.nominal.arff"));
		Instances data = new Instances (breader);
		data.setClassIndex(data.numAttributes() - 1);
		MyID3withGainRatio decision_tree = new MyID3withGainRatio();
		decision_tree.buildClassifier(data);
		
		for (int i = 0; i < data.numInstances(); i++) {
			System.out.println(data.instance(i) + " : " + data.instance(i).classValue());
		}
		System.out.println();
		
		decision_tree.makeTree(data);
		
	}

}
