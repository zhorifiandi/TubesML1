package algorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import algorithm.C45Support.SetOfRuleForEvaluation;
import structures.Edge;
import structures.Rule;
import structures.SetOfRule;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.Sourcable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import static structures.SetOfRule.convertTreeIntoRules;

public class MyC45 extends Classifier 
implements TechnicalInformationHandler, Sourcable {
	
	  /** The decision tree */
    private MyID3 decision_tree;
	private SetOfRule set_of_rule;
	private Instances train_data;

	public MyID3 getDecision_tree(){
	    return decision_tree;
    }

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

		// Infer the decision tree from the training set, growing the tree
		// until the training data is fit as well as possible and
		// allowing overfitting to occur.
        decision_tree = new MyID3();
		decision_tree.buildClassifier(data);

//		System.out.println(decision_tree.toString());

        train_data = data;

        set_of_rule = convertTreeIntoRules(decision_tree);


        // DONT DELETE THISSSS
        set_of_rule.setTrainData(data);

        List<Rule> listrule = set_of_rule.getList_rule();
        System.out.println("\n\n\nBEFORE PRUNEDDD");
        for (int i = 0; i < listrule.size(); i++ ){
            System.out.println("Rule "+i+" :");
            Rule rule = listrule.get(i);
            List<Edge> edges = rule.getPreconditions();
            for (int j=0; j< edges.size(); j++) {
                System.out.println("\t-"+edges.get(j).getAttribute_name() + " = " + edges.get(j).getAttribute_value());
            }
            System.out.println("\tClass = "+rule.getClass_value());
        }

//        selectBestSetofRule();

	}


    private void selectBestSetofRule() throws Exception {
        SetOfRule rules = set_of_rule;
        // Count Error for Default
        SetOfRuleForEvaluation sorfe = new SetOfRuleForEvaluation();
        sorfe.buildClassifier(train_data);
        sorfe.setSetOfRule(rules);

        int trainSize = Math.round(train_data.numInstances() * 75/ 100);
        int testSize = train_data.numInstances() - trainSize;
        Instances train = new Instances(train_data, 0, trainSize);
        Instances test = new Instances(train_data, trainSize, testSize);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(sorfe,test);

        double default_error = eval.errorRate();
        System.out.println("default : "+default_error);


        for (int i=0 ; i< rules.getList_rule().size(); i++){
            Rule rule = rules.getList_rule().get(i);
            if (rule.getPreconditions().size() > 1) {
                for (int j = 0; j < rule.getPreconditions().size(); j++) {
                    Edge precond = rule.getPreconditions().get(j);
                    SetOfRule temp = rules;
                    temp.getList_rule().get(i).delPrecondition(precond);

                    // Count Error for Default
                    SetOfRuleForEvaluation sorfe_temp = new SetOfRuleForEvaluation();
                    sorfe_temp.buildClassifier(train_data);
                    sorfe_temp.setSetOfRule(temp);

                    int trainSize2 = Math.round(train_data.numInstances() * 75 / 100);
                    int testSize2 = train_data.numInstances() - trainSize;
                    Instances train2 = new Instances(train_data, 0, trainSize2);
                    Instances test2 = new Instances(train_data, trainSize, testSize2);

                    Evaluation eval2 = new Evaluation(train2);
                    eval2.evaluateModel(sorfe_temp, test2);

                    double temp_error = eval2.errorRate();
                    temp.setError(temp_error);


                    System.out.println("Compare\ndefault : "+default_error);
                    System.out.println("temp : "+temp_error);
                    if (temp_error < default_error) {

                        System.out.println("Masuk ");
                        set_of_rule = temp;
                    }
                }
            }
        }




//		int index = 0;
//		for (int i=0; i<setrule.size(); i++){
//            if (setrule.get(i).getError() < setrule.get(index).getError()){
//                index = i;
//            }
//            List<Rule> listrule = setrule.get(i).getList_rule();
//
//            System.out.println("\n\n\nAFTER PRUNEDDD "+i);
//            for (int k = 0; k < listrule.size(); k++ ){
//                System.out.println("Rule "+k+" :");
//                Rule rule = listrule.get(k);
//                List<Edge> edges = rule.getPreconditions();
//                for (int j=0; j< edges.size(); j++) {
//                    System.out.println("\t-"+edges.get(j).getAttribute_name() + " = " + edges.get(j).getAttribute_value());
//                }
//                System.out.println("\tClass = "+rule.getClass_value());
//            }
//        }
//
//        System.out.println("Best : "+ index);



//        SetOfRule result = setrule.get(0);
//		return rules;
	}

//	private List<SetOfRule> generateAllPrunedSetofRule(SetOfRule rules) throws Exception {
//        List<SetOfRule> result = new ArrayList<SetOfRule>();
//
//
//        // Count Error for Default
//        SetOfRuleForEvaluation sorfe = new SetOfRuleForEvaluation();
//        sorfe.buildClassifier(train_data);
//        sorfe.setSetOfRule(rules);
//
//        int trainSize = Math.round(train_data.numInstances() * 75/ 100);
//        int testSize = train_data.numInstances() - trainSize;
//        Instances train = new Instances(train_data, 0, trainSize);
//        Instances test = new Instances(train_data, trainSize, testSize);
//
//        Evaluation eval = new Evaluation(train);
//        eval.evaluateModel(sorfe,test);
//
//        double default_error = eval.errorRate();
//        SetOfRule default_rules = rules;
//        default_rules.setError(default_error);
//        result.add(default_rules);
//
//
//        for (int i=0 ; i< rules.getList_rule().size(); i++){
//            Rule rule = rules.getList_rule().get(i);
//            if (rule.getPreconditions().size() > 1) {
//                for (int j = 0; j < rule.getPreconditions().size(); j++) {
//                    Edge precond = rule.getPreconditions().get(j);
//                    SetOfRule temp = rules;
//                    temp.getList_rule().get(i).delPrecondition(precond);
//
//                    // Count Error for Default
//                    SetOfRuleForEvaluation sorfe_temp = new SetOfRuleForEvaluation();
//                    sorfe_temp.buildClassifier(train_data);
//                    sorfe_temp.setSetOfRule(temp);
//
//                    int trainSize2 = Math.round(train_data.numInstances() * 75 / 100);
//                    int testSize2 = train_data.numInstances() - trainSize;
//                    Instances train2 = new Instances(train_data, 0, trainSize2);
//                    Instances test2 = new Instances(train_data, trainSize, testSize2);
//
//                    Evaluation eval2 = new Evaluation(train2);
//                    eval2.evaluateModel(sorfe_temp, test2);
//
//                    double temp_error = eval2.errorRate();
//                    temp.setError(temp_error);
//
//
//                    if (temp_error < default_error) {
//                        result.add(temp);
//                    }
//                }
//            }
//        }
//
//		return result;
//	}




	@Override
	public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
	    double result = set_of_rule.classify(instance);
	    if (result == -1.0){
            return decision_tree.getMost_common_value();
        }
        else {
            return result;
        }
		
	}

//	public static void main(String[] args) throws Exception {
//        BufferedReader breader = new BufferedReader(new FileReader("arff//weather.nominal.arff"));
//        Instances data = new Instances (breader);
//        data.setClassIndex(data.numAttributes() - 1);
//        MyC45 c45 = new MyC45();
//
//        NumericToNominal filter = new NumericToNominal();
//        Instances filterRes;
//
//        //Algoritma
//        filter.setInputFormat(data);
//        filterRes = Filter.useFilter(data, filter);
//        c45.buildClassifier(filterRes);
//
//        SetOfRule init_set_of_rule = convertTreeIntoRules(c45.getDecision_tree());
//        List<Rule> listrule = init_set_of_rule.getList_rule();
//
//        System.out.println("\n\n\nAFTER PRUNEDDD");
//        for (int i = 0; i < listrule.size(); i++ ){
//            System.out.println("Rule "+i+" :");
//            Rule rule = listrule.get(i);
//            List<Edge> edges = rule.getPreconditions();
//            for (int j=0; j< edges.size(); j++) {
//                System.out.println("\t-"+edges.get(j).getAttribute_name() + " = " + edges.get(j).getAttribute_value());
//            }
//            System.out.println("\tClass = "+rule.getClass_value());
//        }
//
//        System.out.println(data.instance(0) + " : " + data.instance(0).classValue());
//        System.out.println("Predict: "+ init_set_of_rule.classify(data.instance(0)));
//
//	}

}
