package structures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import algorithm.MyID3;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class SetOfRule implements Serializable{
	private List<Rule> list_rule;
	private double accuracy;
	private double error;
	private Instances train_data;
	
	public SetOfRule(){
		list_rule = new ArrayList<Rule>();
		accuracy = 0;
	}

	public void setList_rule(List<Rule> L){
		list_rule = null;
		list_rule = L;
	}

	public List<Rule> getList_rule(){
		return list_rule;
	}

	public void setTrainData(Instances data){
		train_data = data;
	}

	public void setError(double er){
		error = er;
	}
	
	public double getAccuracy() {
		return accuracy;
	}
	
	public void addRule(Rule rule){
		list_rule.add(rule);
	}
	
	public void delRule(Rule rule){
		list_rule.remove(rule);
	}
	
	private void countAccuracy(){
		accuracy = 0;
	}
	
	public double classify(Instance instance){
		boolean classified = true;
		double classValue = -1;
		for (int i = 0; i < list_rule.size(); i ++){
			List<Edge> edges = list_rule.get(i).getPreconditions();
			for (int j = 0; j < edges.size(); j++){
				String att_name = edges.get(j).getAttribute_name();
				double att_value_double = edges.get(j).getAttribute_valueDouble();
				int att_index = edges.get(j).getAttribute_index();

				if (instance.value(att_index) != (att_value_double)){
					classified = false;
					break;
				}
			}

			if (classified){
				classValue = list_rule.get(i).getClass_value();
				break;
			}
		}

//		System.out.println(">>>> "+instance + " ----> "+classValue);

		return classValue;
	}

	private static List<Edge> buffer_edge;
	private static SetOfRule rule_container;

	private static int generateRule(MyID3 decision_tree){
		if (decision_tree != null) {
			Attribute weka_att = decision_tree.getNode_Attribute();

			if (weka_att != null) {
				MyID3[] successors = decision_tree.getNode_Successors();
				for (int i = 0; i < successors.length; i++) {
					Edge att = new Edge();
					att.setAttribute_name(weka_att.name());
					att.setAttribute_value(weka_att.value(i));
					att.setAttribute_valueDouble(weka_att.indexOfValue(weka_att.value(i)));
					att.setAttribute_index(weka_att.index());

					// System.out.println("Before Add1 : " + buffer_edge);
					buffer_edge.add(att);
					// System.out.println("After Add1 : " + buffer_edge);
					generateRule(successors[i]);
				}
				return 1;

			} else {
				// Node Attribute is null => It's a leaf node!
				// One rule generated
				Rule rule = new Rule();
				for (int i = 0; i < buffer_edge.size(); i++) {
					rule.addPrecondition(buffer_edge.get(i));
				}
				rule.setClass_value(decision_tree.getNode_ClassValue());
				rule_container.addRule(rule);

				// System.out.println("Pop1 : " + buffer_edge);
				buffer_edge.remove(buffer_edge.size() - 1);
				return 0;
			}
		}
		else {
			return 0;
		}
	}

	private static void generateRules(MyID3 decision_tree2){
		Attribute weka_att = decision_tree2.getNode_Attribute();

		MyID3[] successors = decision_tree2.getNode_Successors();
		for (int i = 0; i < successors.length; i++) {
			Edge att = new Edge();
			att.setAttribute_name(weka_att.name());
			att.setAttribute_value(weka_att.value(i));
			att.setAttribute_index(weka_att.index());
			att.setAttribute_valueDouble(weka_att.indexOfValue(weka_att.value(i)));

			// System.out.println("Before Add0 : "+ buffer_edge);
			buffer_edge.add(att);
			// System.out.println("After Add0 : "+ buffer_edge);
			int flag = generateRule(successors[i]);
			if (flag == 1) {
				// System.out.println("Pop0 : " + buffer_edge);
				buffer_edge.remove(buffer_edge.size() - 1);
			}
		}

	}

	public static SetOfRule convertTreeIntoRules(MyID3 decision_tree2) {
		buffer_edge = new ArrayList<Edge>();
		rule_container = new SetOfRule();

		generateRules(decision_tree2);


		return rule_container;
	}



	public double getError() {
		return error;
	}

	private double getMostCommonValue(Instances data) {
		int[] counter = new int[data.numClasses()];
		double[] target = new double[data.numClasses()];
		Enumeration en_val = data.classAttribute().enumerateValues();
		int k = 0;
		while (en_val.hasMoreElements()){
			Object temp = en_val.nextElement();
			target[k] = (double) k;
			k++;
		}

		for (int i = 0; i < data.numInstances(); i++){
			for (int j = 0; j < target.length; j++){
				if (data.instance(i).classValue() == target[j]){
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


	private int countMinority(){
		int count = 0;
		double common_value = getMostCommonValue(train_data);
		for (int i=0; i<train_data.numInstances(); i++){
			if (train_data.instance(i).classValue() != common_value){
				count++;
			}
		}

		return count;
	}

	public void countError(){
		double cf = 0.95;
		int minority = countMinority();
		double binomial_result = binomial(cf,minority,train_data.numInstances());
		error = train_data.numInstances() * binomial_result;
	}

	public static double binomial( double p, int k, int N) {
		double[][] b = new double[N+1][k+1];

		// base cases
		for (int i = 0; i <= N; i++)
			b[i][0] = Math.pow(1.0 - p, i);
		b[0][0] = 1.0;

		// recursive formula
		for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= k; j++) {
				b[i][j] = p * b[i-1][j-1] + (1.0 - p) *b[i-1][j];
			}
		}
		return b[N][k];
	}

	public static void main(String[] args) throws Exception {
		BufferedReader breader = new BufferedReader(new FileReader("arff//weather.nominal.arff"));
		Instances data = new Instances (breader);
		data.setClassIndex(data.numAttributes() - 1);
		MyID3 decision_tree = new MyID3();
		decision_tree.buildClassifier(data);

		SetOfRule init_set_of_rule = convertTreeIntoRules(decision_tree);
		List<Rule> listrule = init_set_of_rule.getList_rule();

		for (int i = 0; i < listrule.size(); i++ ){
			System.out.println("Rule "+i+" :");
			Rule rule = listrule.get(i);
			List<Edge> edges = rule.getPreconditions();
			for (int j=0; j< edges.size(); j++) {
				System.out.println("\t-"+edges.get(j).getAttribute_name() + " = " + edges.get(j).getAttribute_value());
			}
			System.out.println("\tClass = "+rule.getClass_value());
		}

		System.out.println(data.instance(0) + " : " + data.instance(0).classValue());
		System.out.println("Predict: "+ init_set_of_rule.classify(data.instance(0)));
	}
}
