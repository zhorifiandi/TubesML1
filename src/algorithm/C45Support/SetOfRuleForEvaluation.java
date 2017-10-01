package algorithm.C45Support;

import structures.SetOfRule;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;

import java.util.Enumeration;

/**
 * Created by zhorifiandi on 10/1/17.
 */
public class SetOfRuleForEvaluation extends Classifier {
    private SetOfRule setOfRule;
    private Instances train_data;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        train_data = instances;
    }

    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        double result = setOfRule.classify(instance);
        if (result == -1.0){
            return getMostCommonValue(train_data);
        }
        else {
            return result;
        }

    }

    protected double getMostCommonValue(Instances data) {
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

    public SetOfRule getSetOfRule() {
        return setOfRule;
    }

    public void setSetOfRule(SetOfRule setOfRule) {
        this.setOfRule = setOfRule;
    }
}
