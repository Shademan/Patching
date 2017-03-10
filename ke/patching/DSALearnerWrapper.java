/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tud.ke.patching;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Puk;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

/**
 * Wrapper class for explicit learning the decision regions which are classified
 * falsely. The actual classifier will be provided a binary class problem
 * (1=failure, 0=normal) If possible, and the classifier is enabled to do this
 * (e.g. ExtJRip), it can provide further information (an integer) on WHERE in
 * the decision space an instance lies.
 *
 * @author SKauschke
 */
public class DSALearnerWrapper extends AbstractClassifier
        implements DeciderEnumerator {

    private int lastUsedDecider = 0;

    
    private Classifier classifier;
    private Boolean isBuilt = false;

    public DSALearnerWrapper() {
        this.classifier = new JRip();
    }

    public DSALearnerWrapper(Classifier classy) {

        this.classifier = classy;
    }

    @Override
    public void buildClassifier(Instances data) {

        try {
            this.classifier.buildClassifier(data);
            this.isBuilt = true;
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    @Override
    public double classifyInstance(Instance a) throws Exception {

        double label = -1;
        if (isBuilt) {
            label = classifier.classifyInstance(a);

            if (classifier instanceof DeciderEnumerator) {
                DeciderEnumerator de = (DeciderEnumerator) classifier;
                this.lastUsedDecider = de.getLastUsedDecider();
            } else {
                if(label>0) this.lastUsedDecider = 1;
                else this.lastUsedDecider = 0;
            }
        }
        return label;
    }

    /**
     * Returns the total amount of deciders that exist (means: amount of rules,
     * or amount of leafs in the decision tree)
     *
     * @return
     */
    public int getAmountOfDeciders() {

        if (classifier instanceof DeciderEnumerator) {
            DeciderEnumerator de = (DeciderEnumerator) classifier;
            return de.getAmountOfDeciders();
        }

        return 1;
    }

    /**
     * Returns the number of the decider that was responsible for the last
     * instance that was classified
     *
     * @return
     */
    public int getLastUsedDecider() {

        return this.lastUsedDecider;    // is computed in classifyInstance!!!!
    }

    /**
     * Returns the number of the "default rule" which covers all previously
     * unclassified instances. In case of decision trees, i dont know what this
     * should do :D probably return -1 or so.
     *
     * @return
     */
    public int getDefaultDecider() {
        
        if (classifier instanceof DeciderEnumerator) {
            DeciderEnumerator de = (DeciderEnumerator) classifier;
            return de.getDefaultDecider();
        }
        
        return 0;
    }

    public int getRegionId(Instance a) {
        
        if(this.classifier instanceof ExtRip) {
            ExtRip er = (ExtRip) classifier;
            return er.getLastUsedDecider();
        }
        
        try {
            return (int) classifyInstance(a);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }

        return -1;
    }
    
    @Override
    public String toString() {
        return this.classifier.toString();
    }
}
