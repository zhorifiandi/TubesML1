package algorithm.C45Support;

import java.util.Enumeration;

import algorithm.MyID3;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

public class MyID3withGainRatio extends MyID3 {
	private MyID3withGainRatio[] node_Successors;
	
//	private double computeInfoGain(Instances data, Attribute att) 
//	   throws Exception {
//
//	   double infoGain = computeEntropy(data);
//	   Instances[] split_data = splitData(data, att);
//	   for (int j = 0; j < att.numValues(); j++) {
//	     if (split_data[j].numInstances() > 0) {
//	       infoGain -= ((double) split_data[j].numInstances() /
//	                    (double) data.numInstances()) *
//	         computeEntropy(split_data[j]);
//	     }
//	   }
//	   return infoGain;
//	 }

//	^^^ Contoh, itu yg ID3 biasa 
	
	
	private double computeInfoGain(Instances data, Attribute att) 
			   throws Exception {
			//PAKE GAIN RATIO
			//IMPLEMENT INI FAIQ
			//Jangan lupa handle kalo continuous value
			
			/*
			 * Pseudo code:
			 * 
			 * for each attribute:
			 * 		if (att.type == Attribute.NUMERIC) {
			 * 			rumus ngitung continue
			 * 		}
			 * 		else {
			 * 			GainRatio di buku / slide
			 * 		}
			 * 
			 * 
			 */
			
			return 0;
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

}
