/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tud.ke.patching;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import java.util.Iterator;
import java.util.Vector;
import moa.classifiers.AbstractClassifier;
import moa.options.WEKAClassOption;
import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

/**
 *
 * @author SKauschke
 */
public class AdaptivePatching extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    private Classifier baseClassifier;
    protected SamoaToWekaInstanceConverter instanceConverter = new SamoaToWekaInstanceConverter();

    int updates = 0;    // how many update phases have been executed
    int numInstances = 0;  // how many instances have been seen
    int instancesInBatch = 0; // how many instances have been seen in this batch

    DSALearnerWrapper regionDecider;
    Vector regionPatches;

    protected weka.core.Instances instancesBuffer;

    Instances origData;
    Instances errorInstances;
    Instances reDefinedClasses;
    InstanceStore instanceStore;

    Boolean initPhase = true;
    Vector subsets = new Vector();
    double basePerformance = 0;
    Vector basePerfOnSubset = new Vector();

    public AdaptivePatching() {
        super();
    }

    @Override
    public String getPurposeString() {
        return "Adaptive Patching for the classification of evolving data streams.";
    }

    /**
     * Options that can be adjusted via the MOA interface
     */
    public IntOption initialBatchSize = new IntOption("InitialBucketSize", 'i',
            "Size of the first batch that is used to create the base model.", 5000,
            0, Integer.MAX_VALUE);

    public IntOption batchSize = new IntOption("batchSize", 'b',
            "The number of instances to observe between model updates.", 500,
            0, Integer.MAX_VALUE);

    public IntOption batchesToKeep = new IntOption("batchesToKeep", 'k',
            "The number of batches to keep in the Instance Store.", 5,
            0, Integer.MAX_VALUE);

    public FlagOption forceNoAdaptation = new FlagOption("forceNoAdaptation", 'f', "If set, NO adaptation is processed!");

    public WEKAClassOption baseClassifierOption = new WEKAClassOption("baseLearner", 'l',
            "WEKA class to use for the base classifier.", weka.classifiers.Classifier.class, "weka.classifiers.rules.JRip");

    public WEKAClassOption dsClassifierOption = new WEKAClassOption("decisionSpaceLearner", 'd',
            "WEKA class to use for learning the decision space in which the errors lie.", weka.classifiers.Classifier.class, "weka.classifiers.rules.JRip");//"moa.tud.ke.patching.ExtJRip");

    public WEKAClassOption patchClassifierOption = new WEKAClassOption("patchLearner", 'p',
            "WEKA class to use for the patches as classifier.", weka.classifiers.Classifier.class, "weka.classifiers.rules.JRip");

    /**
     * Resets the learning process completely, as if nothing was ever learned.
     */
    public void resetLearningImpl() {

        this.baseClassifier = null;

        this.updates = 0;
        this.numInstances = 0;
        this.instancesInBatch = 0;

        this.origData = null;
        this.errorInstances = null;
        this.reDefinedClasses = null;
        this.subsets = new Vector();
        this.basePerformance = 0;
        this.basePerfOnSubset = new Vector();
        this.regionPatches = new Vector();
        this.regionDecider = null;
        this.instanceStore = new InstanceStore(batchesToKeep.getValue());

        initPhase = true;
    }

    /**
     * Trains on a new instance
     *
     * @param inst
     */
    public void trainOnInstanceImpl(Instance samoaInstance) {

        // Fill the new instance into the buffer...
        weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        if (instancesBuffer != null) {
            this.instancesBuffer.add(inst);
        } else {
            weka.core.Instances tmp = new weka.core.Instances(inst.dataset());
            this.instancesBuffer = tmp;
        }

        this.numInstances++;
        this.instancesInBatch++;

        // React according to which phase we are in
        if (this.initPhase) {    // Initialisation phase

            if (instancesInBatch > this.initialBatchSize.getValue()) {
                this.initPhase = false;

                this.origData = instancesBuffer;
                instancesBuffer = null; // kill teh buffa
                buildBaseClassifier();
                instancesInBatch = 0; // reset this
            }

        } else { // Batch acquisition + update phase

            if (instancesInBatch >= batchSize.getValue()) {

                // Update the classifier if allowed
                if (!forceNoAdaptation.isSet()) {
                    updateClassifier(instancesBuffer);
                }

                // and reset the instanceBuffer
                instancesInBatch = 0; // reset
                this.instancesBuffer = null;
            }
        }

    }

    /**
     * Returns an instance of the trained base classifier.
     */
    private void buildBaseClassifier() {
        try {
            this.baseClassifier = getBaseClassifier();
            this.baseClassifier.buildClassifier(origData);
        } catch (Exception e) {
            System.err.println("Error building base classifier:");
            System.err.println(e.getMessage());
        }
    }

    /**
     * Returns a classifier object specified by the options given in the MOA UI
     *
     * @return
     */
    private Classifier getBaseClassifier() {
        try {
            String[] options = weka.core.Utils.splitOptions(baseClassifierOption.getValueAsCLIString());
            Classifier tmp = createWekaClassifier(options);

            return tmp;
        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.baseClassifierOption.getValueAsCLIString());
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Returns a classifier object specified by the options given in the MOA UI
     *
     * @return
     */
    private Classifier getPatchClassifier() {
        try {
            String[] options = weka.core.Utils.splitOptions(baseClassifierOption.getValueAsCLIString());
            Classifier tmp = createWekaClassifier(options);

            return tmp;
        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.baseClassifierOption.getValueAsCLIString());
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Returns a classifier object specified by the options given in the MOA UI
     *
     * @return
     */
    private DSALearnerWrapper getDecisionSpaceClassifier() {
        try {
            String[] options = weka.core.Utils.splitOptions(dsClassifierOption.getValueAsCLIString());
            Classifier tmp = createWekaClassifier(options);

            if (tmp instanceof JRip) {
                //System.out.println("CHANGING JRip to ExtRip");
                tmp = new ExtRip();
            }

            return new DSALearnerWrapper(tmp);
        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.dsClassifierOption.getValueAsCLIString());
            System.err.println(e.getMessage());
        }
        return null;
    }

    public void updateClassifier(Instances data) {

        this.updates++;

        // First: merge the new instances to the "Instance Store"
        this.instanceStore.addInstances(data);

        System.out.println("Update at Instance: " + this.numInstances + " | Size of Instance store (updates:" + this.updates + "): " + instanceStore.getInstances().size());

// Turn the instances into a binary learning problem to learn the decision space where the original classifier was wrong
        this.reDefinedClasses = redefineProblem();

        // Now: learn the error regions with a specially adapted or a normal classifier:
        try {
            this.regionDecider = new DSALearnerWrapper(getDecisionSpaceClassifier());
            regionDecider.buildClassifier(reDefinedClasses);
        } catch (Exception e) {
            System.err.println("Error building region decider");
            System.err.println(e.getStackTrace());
            System.err.println(e.getMessage());
            System.exit(123452345);
        }

        // Determine the subsets of instances which are covered by the rules (that are not the default rule)
        this.subsets = determineSubsets(data, regionDecider);
        System.out.println("Region Decision Subsets: " + subsets.size());

        // Determine the performance of the BASE classifier for each of those subsets
        //this.basePerfOnSubset = determineBasePerformanceOnSubsets(this.subsets, baseClassifier);
        // Create individual models for the subsets
        this.regionPatches = createPatches(this.subsets, this.basePerfOnSubset);

    }

    /**
     * Merges two sets of instances
     *
     * @param a
     * @param b
     * @return
     */
    private Instances mergeInstances(Instances a, Instances b) {
        Instances merged = a;
        Iterator it = b.iterator();
        while (it.hasNext()) {
            weka.core.Instance i = (weka.core.Instance) it.next();
            merged.add(i);
        }
        return merged;
    }

    /**
     * Learns a specific subset classifier (of the same type as the base
     * classifier) to improve accuracy on the regions that performed bad before.
     *
     * @param subsets
     * @param basePerformance
     * @return
     */
    private Vector createPatches(Vector subsets, Vector basePerformance) {
        Vector patches = new Vector();

        //System.out.println("Creating patches: #"+subsets.size());
        try {
            for (int d = 0; d < subsets.size(); d++) {

                Instances set = (Instances) subsets.get(d);

                Classifier patch;
                if (set.size() < 5) // Too small to do anything properly
                {
                    patch = new ZeroR();
                } else {
                    patch = getPatchClassifier();
                }
                patch.buildClassifier(set);
                patches.add(d, patch);
            }
        } catch (Exception e) {
            System.err.println("Error building patches:");
            System.err.println(e.getMessage());
        }
        return patches;
    }

    /**
     * Quickly calculate accuracy for all subsets with the base classifier.
     *
     * @param subsets
     * @param base
     * @return
     */
    private Vector determineBasePerformanceOnSubsets(Vector subsets, Classifier base) {
        Vector perf = new Vector();

        for (int d = 0; d < subsets.size(); d++) {

            Instances sub = (Instances) subsets.get(d);
            perf.add(d, determinePerformance(sub, base));
        }

        return perf;
    }

    private Vector determineSubsets(Instances data, Classifier detector) {
        Vector subsets = new Vector();

        if (detector instanceof DeciderEnumerator) {

            DeciderEnumerator decider = (DeciderEnumerator) detector;

            int numDeciders = decider.getAmountOfDeciders();
            int lastDecider = 0;

            int d = 0;
            if (numDeciders == 1) {  // ACHTUNG: wenn nur ein Decider da ist (also keine Subunterteilung der decision spaces vorliegt)
                d = 1;              // dann wird hier die parametrisierung für die folgende schleife angepasst.
                numDeciders++;
            }

            // In order to save some ram we are going to do this iteratively
            while (d < numDeciders) {
                Instances copy = new Instances(data); // Make a clone of the full dataset

                // Iterate over all instances, classify them and delete all instances 
                // that do not belong to the current decider from the dataset
                Iterator i = copy.iterator();
                try {
                    while (i.hasNext()) {
                        weka.core.Instance inst = (weka.core.Instance) i.next();

                        detector.classifyInstance(inst);
                        lastDecider = decider.getLastUsedDecider();
                        if (lastDecider != d) {
                            i.remove();
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Something went wrong while trying to split into subsets:");
                    System.err.println(e.getMessage());
                }

                subsets.add(copy);
                d++;
            }

            int defaultDecider = decider.getDefaultDecider();
            if (decider.getAmountOfDeciders() > 1 && defaultDecider >= 0) {
                subsets.remove(defaultDecider);
            }

        }

        return subsets;
    }

    private double determinePerformance(Instances data, Classifier classif) {

        double numInstances = data.numInstances();
        double correctInstances = 0;
        double klasse;
        Double accuracy;

        try {
            Iterator in_it = data.iterator();
            while (in_it.hasNext()) {
                weka.core.Instance ins = (weka.core.Instance) in_it.next();
                klasse = baseClassifier.classifyInstance(ins);

                if (klasse == ins.classValue()) {
                    correctInstances++;
                }
            }
            accuracy = new Double(correctInstances / numInstances);

            return accuracy;

        } catch (Exception e) {
            System.err.println("Something went wrong while trying to classify the data");
            System.err.println(e.getMessage());
        }

        return 0;
    }

    /**
     * Creates a copy of the instances and removes all that have been correctly
     * classified by the current model. Saves the error instances to
     * this.errorInstances (accumulated over the various batches)
     */
    private void determineErrors(Instances data) {

        Instances batchError = new Instances(data); // deep copy

        int removed = 0;

//        System.out.println("Before filtering: "+errorInstances.size());
        double predictedClass = 0;

        Iterator inst = batchError.iterator();
        while (inst.hasNext()) {
            weka.core.Instance a = (weka.core.Instance) inst.next();
            try {
                predictedClass = this.classifyInstance(a);
            } catch (Exception e) {
                System.err.println("Error while classifying instance in determineErrors.");
                System.err.println(e.getMessage());
            }

            if (predictedClass == a.classValue()) {
                inst.remove();  // remove from the errorInstances
                removed++;
            }
        }

        this.errorInstances = batchError;

    }

    /**
     * Creates a copy (switchClass) of the instances and redefines the problem
     * such that it is now important to classify the wrongly classified
     * instances
     */
    private Instances redefineProblem() {

        Instances redefInstances = new Instances(this.instanceStore.getInstances()); // deep copy of instance store

//        System.out.println(reDefinedClasses.attributeStats(reDefinedClasses.classIndex()));
//        System.out.println("Before filtering: "+wrongData.size());
        double predictedClass = 0;

        Iterator inst = redefInstances.iterator();
        while (inst.hasNext()) {
            weka.core.Instance a = (weka.core.Instance) inst.next();
            try {
                predictedClass = this.baseClassifier.classifyInstance(a); // Achtung: das hier muss "base" bleiben!!
            } catch (Exception e) {
                System.err.println("Error while classifying instance in redefineProblem");
                System.err.println(e.getMessage());
            }

            if (predictedClass == a.classValue()) {
                a.setClassValue(0);
            } else {
                a.setClassValue(1);
            }
        }

        return redefInstances;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        out.append("Uses a base classifier to create a default classifier, and then extends it when necessary with patches of that same type of classifier.");
    }

    public double classifyInstance(weka.core.Instance inst) throws Exception {

        int region = -1;
        int defaultDecider = -1;
        int amountDeciders = -1;
        double label;

        try {
            if (this.regionDecider != null) {
                // Pre-classify instance and retrieve the used Decider
                double regClass = this.regionDecider.classifyInstance(inst);

                region = this.regionDecider.getLastUsedDecider();
                defaultDecider = this.regionDecider.getDefaultDecider();
                amountDeciders = this.regionDecider.getAmountOfDeciders();

                Classifier patch;

                if (region != defaultDecider) {
                    if (amountDeciders > 1) {       // case a: there are actual region deciders
                        patch = (Classifier) regionPatches.elementAt(region);
                        //System.out.println("using patch region decider: "+region);
                        label = patch.classifyInstance(inst);
                        return label;
                    } else {                        // case b: we only have a 0/1 information about if its in the error region or not.
                        patch = (Classifier) regionPatches.elementAt(0);
                        label = patch.classifyInstance(inst);
                        return label;
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("AdaptivePatching : Error in classifyInstance while using regionDecider.");
            System.out.println("Region: " + region + " DefaultDecider:" + defaultDecider + " amountDeciders:" + amountDeciders + " regionPatches#:" + regionPatches.size());
            e.printStackTrace();
            System.exit(234545345);
        }

        return baseClassifier.classifyInstance(inst);
    }

    /**
     * classify an instance
     *
     * @param inst
     * @return
     */
    public double[] getVotesForInstance(Instance samoaInstance) {

        weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        int numClasses = inst.attribute(inst.classIndex()).numValues();

        double[] votes = new double[numClasses];
        //System.out.println("Num Values: "+numAttributes);

        for (int i = 0; i < numClasses; i++) {
            votes[i] = 0;
        }

        if (this.baseClassifier != null) {
            try {
                int klasse = (int) Math.round(classifyInstance(inst));
                votes[klasse] = 1;
            } catch (Exception e) {
                System.err.println("Classification failed... pfft.");
            }
        }
        return votes;
    }

    public boolean isRandomizable() {
        return true;
    }

    protected moa.core.Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    /**
     * Helper function: Create a weka classifier based on a string.
     *
     * @param options
     * @throws Exception
     */
    private Classifier createWekaClassifier(String[] options) throws Exception {

        Classifier classy;

        String classifierName = options[0];
        String[] newoptions = options.clone();
        newoptions[0] = "";
        classy = weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);

        return classy;
    }

}