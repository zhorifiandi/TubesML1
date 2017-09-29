package driver;


import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.Scanner;

import algorithm.MyID3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NumericTransform;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.Standardize;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.*;

/**
 *
 * @author user-ari
 */
public class DriverWekaID3 {
    
   static void saveModel(Classifier C, String namaFile) throws Exception {
        //SAVE 
        // serialize model
        String dir = "models//" + namaFile + ".model";
        weka.core.SerializationHelper.write(dir, C);
    }
    
    public static void main(String[] args) throws Exception {
        Scanner scan = new Scanner(System.in);
        BufferedReader breader = null;
        
        //File input
        System.out.print("Masukkan Nama File Input: ");
        String filename = scan.next();
        breader = new BufferedReader(new FileReader("arff//"+ filename +".arff"));
        Instances fileTrain = new Instances (breader);
        breader.close();
  
        
        //Remove if student
        if (filename.equals("student-train") || filename.equals("student-mat-test")){
            
            Remove R = new Remove();
            R.setAttributeIndices("27");
            R.setInputFormat(fileTrain);
            fileTrain = Filter.useFilter(fileTrain, R);
            fileTrain.setClassIndex(26);
            /*
            int columnIndex = fileTrain.classIndex() + 1;
            if (fileTrain.classIndex() != fileTrain.numAttributes() -1){
                String order = "";
                for (int i = 1; i < fileTrain.numAttributes() + 1; i++) {
                  // skip new class
                  if (i == columnIndex)
                    continue;

                  if (!order.equals(""))
                    order += ",";
                  order += Integer.toString(i);
                }
                if (!order.equals(""))
                  order += ",";
                order += Integer.toString(columnIndex);

                // process data
                Reorder reorder = new Reorder();
                reorder.setAttributeIndices(order);
                System.out.println(order);
                reorder.setInputFormat(fileTrain);
                fileTrain = Filter.useFilter(fileTrain, reorder);

                // set class index
                fileTrain.setClassIndex(fileTrain.numAttributes() - 1);
            }*/
        }else{
            
            //Index kelas
            System.out.print("Masukkan index dari atribut yang menjadi kelas "
                    + "(-1 untuk last index): ");
            int classIndex = scan.nextInt();
            if (classIndex == -1) classIndex = fileTrain.numAttributes() - 1;
            fileTrain.setClassIndex(classIndex);

        }
        //Pilih classifier
        System.out.print("Pilih classifier yang akan digunakan (0: NB, 1: FFNN): ");
        int classifierChoice = scan.nextInt();
        Classifier classifier;
        if (classifierChoice == 0){
            classifier = new Id3();
        } else {    
            classifier = new MyID3();
            NumericToNominal ntn = new NumericToNominal();
        	ntn.setInputFormat(fileTrain);
          	fileTrain = Filter.useFilter(fileTrain, ntn);
//            if (!filename.equals("iris")){
//                //Nominal to Binary
//                NominalToBinary ntb = new NominalToBinary();
//                ntb.setInputFormat(fileTrain);
//                fileTrain = Filter.useFilter(fileTrain, ntb);
//                
//                //Standardize
//                Standardize nor = new Standardize();
//                nor.setInputFormat(fileTrain);
//                fileTrain = Filter.useFilter(fileTrain, nor);
//                
//                //NormalizeS
//                Normalize nor1 = new Normalize();
//                nor1.setInputFormat(fileTrain);
//                fileTrain = Filter.useFilter(fileTrain, nor1);
//            }
        }
       
        //Evaluate
        Evaluation eval = new Evaluation(fileTrain);
        boolean validasi = false;
        do {
            System.out.println("\n\n=================\n==== OPTION ====");
            System.out.println("1. Full Training Scheme");
            System.out.println("2. 10 Fold Validation Scheme");
            System.out.println("3. Split Test Scheme (66%-34%)");
            System.out.println("4. Load");
            System.out.println("6. Exit");
            System.out.print("Enter your option (1/2/3/4/5/6): ");
            int pilihan = scan.nextInt();
            switch (pilihan) {
                case 1:
                    {
                        if (classifierChoice == 0){
                            Discretize filter = new Discretize();
                            Instances filterRes;

                            //Algoritma
                            filter.setInputFormat(fileTrain);
                            filterRes = Filter.useFilter(fileTrain, filter);
                            
                            classifier.buildClassifier(filterRes);
                            eval.evaluateModel(classifier, filterRes);
                        } else {
                            classifier.buildClassifier(fileTrain);
                            eval.evaluateModel(classifier, fileTrain);
                        }
                        //OUTPUT
                        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                        System.out.println(eval.toMatrixString("===Confusion matrix==="));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            saveModel(classifier, infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                case 2:
                    {
                        if (classifierChoice == 0){
                            Discretize filter = new Discretize();
                            Instances filterRes;

                            //Algoritma
                            filter.setInputFormat(fileTrain);
                            filterRes = Filter.useFilter(fileTrain, filter);
                            
                            classifier.buildClassifier(filterRes);
                            eval.crossValidateModel(classifier, filterRes, 10, new Random(1));
                        } else {
                            classifier.buildClassifier(fileTrain);
                            eval.crossValidateModel(classifier, fileTrain, 10, new Random(1));
                        }
                        
                        //OUTPUT
                        System.out.println(eval.toSummaryString("=== 10-fold-cross-validation ===\n",true));
                        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                        System.out.println(eval.toMatrixString("===Confusion matrix==="));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            saveModel(classifier, infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                case 3:
                    {
                        int trainSize = (int) Math.round(fileTrain.numInstances() * 66/ 100);
                        int testSize = fileTrain.numInstances() - trainSize;
                        Instances train = new Instances(fileTrain, 0, trainSize);
                        Instances test = new Instances(fileTrain, trainSize, testSize);
                        if (classifierChoice == 0){
                            Discretize filter = new Discretize();
                            Instances filterRes;

                            //Algoritma
                            filter.setInputFormat(fileTrain);
                            filterRes = Filter.useFilter(fileTrain, filter);
                            
                            classifier.buildClassifier(filterRes);
                            eval.evaluateModel(classifier, filterRes);
                        } else {
                            classifier.buildClassifier(train);
                            eval.evaluateModel(classifier, test);
                        }
                        //OUTPUT
                        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                        System.out.println(eval.toMatrixString("===Confusion matrix==="));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            saveModel(classifier, infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       
                        break;
                    }
                case 4:
                    //LOAD
                    // deserialize model
                    System.out.print("Please enter the file name : ");
                    String namaFile = scan.next();
                    Classifier cls = (Classifier) weka.core.SerializationHelper.read("models//" + namaFile + ".model");
                    eval.crossValidateModel(cls, fileTrain, 10, new Random(1));
                    System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                    System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                    System.out.println(eval.toMatrixString("===Confusion matrix==="));
                    System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                    break;
                case 6:
                    validasi = true;
                    break;
                default:
                    System.out.println("Wrong input!");
                    break;
            }
        } while (!validasi);
    }
}