/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tud.ke.patching;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.options.WEKAClassOption;
import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

/**
 *
 * @author SKauschke
 */
public class AdaptivePatchingTwoAdwins extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    private Classifier baseClassifier;
    protected SamoaToWekaInstanceConverter instanceConverter = new SamoaToWekaInstanceConverter();

    protected ExtADWIN ADError;
    protected int maxBatchesToKeep;
    protected int corrected_adwin_size = 0;

    protected boolean changeFine = false;
    protected boolean change = false;

    int updates = 0;    // how many update phases have been executed
    int numInstances = 0;  // how many instances have been seen
    int instancesInBatch = 0; // how many instances have been seen in this batch

    DSALearnerWrapper regionDecider;
    Vector regionPatches;

    protected Instances instancesBuffer;

    Instances origData;
    Instances errorInstances;
    Instances reDefinedClasses;
    Instances prototypeData;
    InstanceStore instanceStore;

    Boolean initPhase = true;
    Vector subsets = new Vector();
    double basePerformance = 0;
    Vector basePerfOnSubset = new Vector();

    public AdaptivePatchingTwoAdwins() {
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

    public FloatOption adwinDelta = new FloatOption("ADWINDelta", 'x',
            "Delta for the ADWIN Algorithm.", 0.002,
            0.001, 100.0);

    public FloatOption fineAdwinDelta = new FloatOption("fineADWINDelta", 'y',
            "Delta for the fine ADWIN Algorithm.", 1,
            0.001, 100.0);

    public FlagOption useBaseClassAsAttribute = new FlagOption("useBaseClassAsAttribute", 'a', "Use the result of the base classifier as additional attribute for the patches");

    public FlagOption forceNoAdaptation = new FlagOption("forceNoAdaptation", 'f', "If set, NO adaptation is processed!");

    public WEKAClassOption baseClassifierOption = new WEKAClassOption("baseLearner", 'l',
            "WEKA class to use for the base classifier.", Classifier.class, "weka.classifiers.rules.JRip");

    public WEKAClassOption dsClassifierOption = new WEKAClassOption("decisionSpaceLearner", 'd',
            "WEKA class to use for learning the decision space in which the errors lie.", Classifier.class, "weka.classifiers.rules.JRip");//"moa.tud.ke.patching.ExtJRip");

    public WEKAClassOption patchClassifierOption = new WEKAClassOption("patchLearner", 'p',
            "WEKA class to use for the patches as classifier.", Classifier.class, "weka.classifiers.rules.JRip");

    public FileOption saveLoadModel = new FileOption("Modelfile", 'm', "Where to store/load the base model from/to", "", ".model", false);

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

        this.ADError = new ExtADWIN(adwinDelta.getValue(), batchSize.getValue());
        this.maxBatchesToKeep = Integer.MAX_VALUE;

        initPhase = true;
    }

    /**
     * Trains on a new instance
     *
     * @param samoaInstance
     */
    public void trainOnInstanceImpl(Instance samoaInstance) {

        // Fill the new instance into the buffer...
        weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        if (this.instancesBuffer != null) {
            this.instancesBuffer.add(inst);
        } else {
            Instances tmp = new Instances(inst.dataset());
            this.instancesBuffer = tmp;
        }

        this.numInstances++;
        this.instancesInBatch++;

        // React according to which phase we are in
        if (this.initPhase) {    // Initialisation phase

            if (instancesInBatch >= this.initialBatchSize.getValue()) {
                this.initPhase = false;

                this.origData = instancesBuffer;
                instancesBuffer = null; // kill teh buffa
                buildBaseClassifier();
                instancesInBatch = 0; // reset this

                if (useBaseClassAsAttribute.isSet()) {
                    this.prototypeData = new Instances(this.origData);  // create an empty instance set, just for the attribute configuration
                    while (prototypeData.size() > 0) {
                        this.prototypeData.delete(0);
                    }
                }
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
     * Returns an instance of the trained base classifier. Also saves/loads the
     * classifier from/to a file if required.
     */
    private void buildBaseClassifier() {
        System.out.println("Building base classifier after " + this.instancesInBatch + " instances.");
        try {
            Boolean baseExists = false;
            if (this.saveLoadModel.getValue().length() > 1) {
                System.out.print("Loading base classifier from " + this.saveLoadModel.getValue() + " ... ");
                this.baseClassifier = FileModel.tryNLoadModel(this.saveLoadModel.getValue());
                if (this.baseClassifier != null) {
                    baseExists = true;
                    System.out.println(" succeeded.");
                } else {
                    System.out.println(" failed (file does not exist)");
                }
            }

            if (!baseExists) {
                this.baseClassifier = getBaseClassifier();
                this.baseClassifier.buildClassifier(origData);

                if (this.saveLoadModel.getValue().length() > 1) {
                    System.out.println("Saving base classifier to " + this.saveLoadModel.getValue());
                    FileModel.saveModel(this.saveLoadModel.getValue(), this.baseClassifier);
                }
            }

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
            String[] options = weka.core.Utils.splitOptions(patchClassifierOption.getValueAsCLIString());
            Classifier tmp = createWekaClassifier(options);

            return tmp;
        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.patchClassifierOption.getValueAsCLIString());
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

            //tmp = new IBk(1);

            return new DSALearnerWrapper(tmp);
        } catch (Exception e) {
            System.err.println("Error retrieving selected classifier:");
            System.err.println("Chosen classifier: " + this.dsClassifierOption.getValueAsCLIString());
            System.err.println(e.getMessage());
        }
        return null;
    }

    public void updateClassifier(Instances data) {

        System.out.println("########## UPDATE PHASE ############");
        this.updates++;

        // Performance berechnen und Adwin befüllen
        System.out.println("Data size: " + data.size());
        System.out.println("Determine Performance...");
        determinePerformance(data, baseClassifier);



        if((this.ADError.getWidth() / batchSize.getValue()) < 1){
            this.instanceStore.setNumBatches(1);
        }else{
            if(change){
                corrected_adwin_size = this.ADError.getWidth() / batchSize.getValue() - 1;
            }
            if (!changeFine) {
                maxBatchesToKeep = this.instanceStore.numBatches;
                corrected_adwin_size++;
            }else{
                maxBatchesToKeep = Math.max(1, this.ADError.getWidth() / batchSize.getValue() - corrected_adwin_size);
            }
            this.instanceStore.setNumBatches(Math.min(maxBatchesToKeep, this.ADError.getWidth() / batchSize.getValue()));
            //this.instanceStore.setNumBatches(this.ADError.getWidth() / batchSize.getValue());
        }

        // wenn die Fenstergröße maximal ist, change wieder auf "false" setzen
        if(this.instanceStore.numBatches == batchesToKeep.getValue()){
            //change = false;
        }

        // First: merge the new instances to the "Instance Store"
        this.instanceStore.addInstances(data);
        System.out.println("size ADWIN: " + this.ADError.getWidth());
        System.out.println("size InstanceStore: " + this.instanceStore.numBatches);
        Instances currentStore = this.instanceStore.getInstances();


        System.out.println("Update at Instance: " + this.numInstances + " | Size of Instance store (updates:" + this.updates + "): " + currentStore.size());

        // Turn the instances into a binary learning problem to learn the decision space where the original classifier was wrong
        //writeArff("C:\\StAtIC\\experiments\\orig.arff", currentStore);

        //writeArff("C:\\StAtIC\\experiments\\modded.arff", this.reDefinedClasses);
//        System.exit(9525356);


        // Determine the subsets of instances which are covered by the rules (that are not the default rule)

        if(this.useBaseClassAsAttribute.isSet())
        {
            currentStore = addBaseClassToInstances(currentStore);
        }

        //if(change) {

            System.out.println("Redefine Problem...");
            this.reDefinedClasses = redefineProblem(currentStore);

            // Now: learn the error regions with a specially adapted or a normal classifier:
            try {
                System.out.println("Build Classifier...");
                this.regionDecider = new DSALearnerWrapper(getDecisionSpaceClassifier());
                regionDecider.buildClassifier(reDefinedClasses);

//            System.out.println("Error Space Classifier:"); System.out.println(regionDecider.toString());       // Todo remove this out
            } catch (Exception e) {
                System.err.println("Error building region decider");
                System.err.println(e.getStackTrace());
                System.err.println(e.getMessage());
                System.exit(123452345);
            }

            System.out.println("Determine Subsets...");
            this.subsets = determineSubsets(currentStore, regionDecider);

            // Determine the performance of the BASE classifier for each of those subsets
            //this.basePerfOnSubset = determineBasePerformanceOnSubsets(this.subsets, baseClassifier);

            // Create individual models for the subsets
            this.regionPatches = createPatches(this.subsets, this.basePerfOnSubset);
            System.out.println("Region Decision Subsets: " + subsets.size());

        //}

//        System.exit(18567820);
        System.out.println("##############################\n\n\n");
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

        System.out.println("Creating patches: #" + subsets.size());
        try {
            for (int d = 0; d < subsets.size(); d++) {

                Instances set = (Instances) subsets.get(d);

//                if(this.useBaseClassAsAttribute.isSet()) {
//                        writeArff("C:\\StAtIC\\experiments\\set"+d+".arff", set);
//                    }
//                System.out.println("Set " + d + " size: " + set.size());
                Classifier patch;
                if (set.size() < 5) // Too small to do anything properly
                {
                    patch = null;   // null will then default to base classifier
                } else {

                    patch = getPatchClassifier();
                    patch.buildClassifier(set);
                }

                patches.add(d, patch);
            }
        } catch (Exception e) {
            System.err.println("Error building patches:");
            System.err.println(e.getMessage());
        }

//        System.out.println("\n--- Patches ------------");
//        for (int i = 0; i < patches.size(); i++) {
//            Classifier tmp = (Classifier) patches.get(i);
//            if (tmp != null) {
//                System.out.print("Patch " + i+" - ");
//                System.out.println(tmp);
//            }
//        }
//        System.out.println("------------------------");
//        System.exit(45768545);
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

    /**
     * Quickly calculate accuracy for all sets with the base classifier.
     *
     * @param sets
     * @param base
     * @return
     */
    private double determineBasePerformance(Instances sets, Classifier base) {
        return determinePerformance(sets, base);
    }

    private Vector determineSubsets(Instances data, Classifier detector) {
        Vector subsets = new Vector();

        if (detector instanceof DeciderEnumerator) {

            DeciderEnumerator decider = (DeciderEnumerator) detector;

            int numDeciders = decider.getAmountOfDeciders();
            int lastDecider = 0;

            Boolean isMultiDecider = true;

            int d = 0;
            if (numDeciders == 1) {  // ACHTUNG: wenn nur ein Decider da ist (also keine Subunterteilung der decision spaces vorliegt)
                d = 1;              // dann wird hier die parametrisierung für die folgende schleife angepasst.
                numDeciders++;
                isMultiDecider = false;
            }

            //System.out.println("Total instances: "+data.size());
            // In order to save some ram we are going to do this iteratively
            while (d < numDeciders) {
                Instances copy = new Instances(data); // Make a clone of the full dataset

                // Iterate over all instances, classify them and delete all instances
                // that do not belong to the current decider from the dataset
                Iterator i = copy.iterator();
                try {
                    while (i.hasNext()) {
                        weka.core.Instance inst = (weka.core.Instance) i.next();

                        double cls = detector.classifyInstance(inst);
                        if (isMultiDecider) {
                            lastDecider = decider.getLastUsedDecider();
                            if (lastDecider != d) {
                                i.remove();
                            } else {
                                if (cls == 1) {
                                    i.remove();  // if the initial classification is correct, we wont bother!!
                                }
                            }
                        } else {
                            if (cls == 1) {
                                i.remove();  // if the initial classification is correct, we wont bother!!
                            }
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Something went wrong while trying to split into subsets:");
                    System.err.println(e.getMessage());
                }

                subsets.add(copy);
                d++;
            }

//            int defaultDecider = decider.getDefaultDecider();
//            if (decider.getAmountOfDeciders() > 1 && defaultDecider >= 0) {
//                subsets.remove(defaultDecider);
//            }
        }

        return subsets;
    }

    private double determinePerformance(Instances data, Classifier classif) {

        double numInstances = data.numInstances();
        double correctInstances = 0;
        double klasse;
        double accuracy;


        change = false;
        changeFine = false;

        try {
            Iterator in_it = data.iterator();
            while (in_it.hasNext()) {
                weka.core.Instance ins = (weka.core.Instance) in_it.next();
                klasse = classif.classifyInstance(ins);

                if (klasse == ins.classValue()) {
                    correctInstances++;
                }

                double estimatedError = this.ADError.getEstimation();
                //System.out.println("estimatedError1: " + estimatedError);
                if(this.ADError.setInput(klasse == ins.classValue() ? 0 : 1, true)){
                    if(this.ADError.getEstimation() > estimatedError){
                        change = true;
                        System.out.println("Change detected!");
                    }else{
                        System.out.println("no change");
                    }
                }

                //System.out.println("estimatedError1: " + estimatedError);
                if(this.ADError.setInput(-1, fineAdwinDelta.getValue(), false)){
                    changeFine = true;
                    System.out.println("Fine Change detected!");
                }

                /*
                System.out.println("Estimation: " + ADError.getEstimation());
                System.out.println("Variance: " + ADError.getVariance());
                System.out.println("Total: " + ADError.getTotal());
                System.out.println("Width: " + ADError.getWidth());
                */

            }

            if(numInstances != 0) {
                accuracy = correctInstances / numInstances;
            }else {
                accuracy = 0;
            }

            return accuracy;

        } catch (Exception e) {
            System.err.println("Something went wrong while trying to classify the data");
            System.err.println(e.getMessage());
        }

        return 0;
    }

    /**
     * Creates a copy of the instances and redefines the problem such that it is
     * now important to classify the wrongly classified instances
     */
    private Instances redefineProblem(Instances data) {

        Instances redefInstances = new Instances(data); // deep copy of instance store

//        System.out.println(reDefinedClasses.attributeStats(reDefinedClasses.classIndex()));
//        System.out.println("Before filtering: "+wrongData.size());
        double predictedClass = 0;

        int oldClassIndex = redefInstances.classIndex();

        try {

            Iterator inst = redefInstances.iterator();
            while (inst.hasNext()) {
                weka.core.Instance a = (weka.core.Instance) inst.next();

                predictedClass = this.baseClassifier.classifyInstance(a); // Achtung: das hier muss "base" bleiben!!

                if (predictedClass == a.classValue()) {
                    a.setClassValue(1);
                } else {
                    a.setClassValue(0);
                }
            }

            if(this.useBaseClassAsAttribute.isSet()) {
                redefInstances = addBaseClassToInstances(redefInstances);
            }

            redefInstances = changeClassToWrongRight(redefInstances);

        } catch (Exception e) {
            System.err.println("Error while classifying instance in redefineProblem");
            System.err.println(e.getMessage());
            System.err.println(e.fillInStackTrace());
            System.exit(987654);
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

        weka.core.Instance origInst = inst;

        try {
            if (this.regionDecider != null) {

                // Handling of optional usage of the base class as an additional attribute.
                if (this.useBaseClassAsAttribute.isSet()) {
                    Instances tmp = new Instances(this.prototypeData); // deep copy of our empty prototypeData
                    tmp.add(inst);
                    tmp = addBaseClassToInstances(tmp);
                    weka.core.Instance inst2 = tmp.get(0);
                    inst = inst2;
                    inst2 = null;
                }

                // Pre-classify instance and retrieve the used Decider
                double regClass = this.regionDecider.classifyInstance(inst);

                if (regClass == 0) {    // only if its in a "wrong" region

                    Boolean isMultiDecider = false;
                    if (this.regionDecider.getAmountOfDeciders() > 1) {
                        isMultiDecider = true;
                    }

                    Classifier patch;

                    if (isMultiDecider) {
                        region = this.regionDecider.getLastUsedDecider();

//                        System.out.println("using patch region decider: "+region);

                        patch = (Classifier) regionPatches.elementAt(region);
                        if (patch != null) {
                            return patch.classifyInstance(inst);
                        }
                    } else {                        // case b: we only have a 0/1 information about if its in the error region or not.
                        patch = (Classifier) regionPatches.elementAt(0);
                        if (patch != null) {
                            return patch.classifyInstance(inst);
                        }

                    }
                } else { // if its not in a "wrong" region, return the class from the base classifier
                    if (this.useBaseClassAsAttribute.isSet()) {
                        return inst.value(0);   // this has maybe already been calculated into the first attribute.
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("AdaptivePatching : Error in classifyInstance while using regionDecider.");
            System.out.println("Region: " + region + " DefaultDecider:" + defaultDecider + " amountDeciders:" + amountDeciders + " regionPatches#:" + regionPatches.size());
            e.printStackTrace();
            System.exit(234545345);
        }

        return baseClassifier.classifyInstance(origInst);
    }

    /**
     * classify an instance
     *
     * @param samoaInstance
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
                System.err.println(e.getMessage());
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

    /**
     * Modify the instances and insert into them the class which the base
     * classifier had them classified as.
     *
     * @return
     */
    private Instances addBaseClassToInstances(Instances origInstances) {

        Instances moddedInstances = new Instances(origInstances); // deep copy

        double predictedClass = 0;

        // create new attribute
        try {
            moddedInstances = copyClassAttribute(moddedInstances, "baseLabel", 1); // das was hier attribute 1 ist, wird zu index 0 
            moddedInstances.setClassIndex(origInstances.classIndex() + 1);
        } catch (Exception e) {
            System.err.println("Error while copying class Attribute for baseLabel");
            System.err.println(e.getMessage());
        }

        Iterator inst = origInstances.iterator();
        int index = 0;
        while (inst.hasNext()) {

            weka.core.Instance a = (weka.core.Instance) inst.next();
            weka.core.Instance target = moddedInstances.instance(index);

            predictedClass = 0;
            try {
                predictedClass = this.baseClassifier.classifyInstance(a); // Achtung: das hier muss "base" bleiben!!
            } catch (Exception e) {
                System.err.println("Error while classifying instance in addBaseClassToInstances");
                System.err.println(a);
                System.err.println(e.getMessage());
            }

            target.setValue(0, predictedClass); // index 0 ist attribute 1 
            index++;
        }

        return moddedInstances;
    }

    /**
     * Copies the class attribute to another position (first position)
     *
     * @param instances
     * @param newName
     * @param newAttributeIndex
     * @return
     * @throws Exception
     */
    public static Instances copyClassAttribute(Instances instances, String newName, int newAttributeIndex) throws Exception {

        int whichAttribute = instances.classIndex();

        Add filter = new Add();
        filter.setAttributeIndex("" + newAttributeIndex);
        filter.setAttributeName(newName);

        // Copy nominal Attribute
        if (instances.attribute(whichAttribute).isNominal()) {
            String newNominalLabels = "";
            Boolean first = true;
            Enumeration<Object> o = instances.attribute(whichAttribute).enumerateValues();
            while (o.hasMoreElements()) {
                String s = (String) o.nextElement();
                if (!first) {
                    newNominalLabels += ",";
                }
                newNominalLabels += s;
                first = false;
            }
            filter.setNominalLabels(newNominalLabels);
        }

        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);
        return instances;
    }

    public static Instances changeClassToWrongRight(Instances instances) throws Exception {

        int whichAttribute = instances.classIndex();

//        System.out.println(instances.classAttribute().toString());
        Add filter = new Add();
        //filter.setAttributeIndex("" + (whichAttribute + 1));
        filter.setAttributeName("newClass");

        String newNominalLabels = "wrong,right";
        filter.setNominalLabels(newNominalLabels);

        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);

        Iterator inst = instances.iterator();
        int index = 0;
        while (inst.hasNext()) {
            weka.core.Instance a = (weka.core.Instance) inst.next();
            a.setValue((whichAttribute + 1), a.classValue());
            index++;
        }

        Remove rmfilter = new Remove();
        rmfilter.setAttributeIndices("" + (instances.classIndex() + 1));
        rmfilter.setInputFormat(instances);
        instances = Filter.useFilter(instances, rmfilter);

        instances.setClassIndex(instances.numAttributes() - 1);

//        System.out.println(instances.classAttribute().toString());
        return instances;
    }

    static void writeArff(String filename, Instances data) {
        try {

            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            writer.write(data.toString());
            writer.flush();
            writer.close();

        } catch (Exception e) {
            System.err.println("Error writing arff file.");
        }
    }

    /**
     * Private class for loading and saving a model to a file.
     */
    private static class FileModel {

        static Classifier tryNLoadModel(String filename) {
            Classifier cls = null;
            try {
                File f = new File(filename);
                if (f.exists() && !f.isDirectory()) {
                    // deserialize model
                    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename));
                    cls = (Classifier) ois.readObject();
                    ois.close();
                }

            } catch (Exception e) {
                System.err.println("Error while loading base model from file: " + filename);
            }
            return cls;
        }

        static void saveModel(String filename, Classifier cls) {

            // Try saving the base model
            try {

//                System.out.println(filename);
//                
//                filename = filename.replace("\\", "\\\\");
                System.out.println(filename);
                //filename = "C:\\StAtIC\\experiments\\mnist\\paramopti\\rf_base.model";

                // serialize model
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
                oos.writeObject(cls);
                oos.flush();
                oos.close();
            } catch (Exception e) {
                System.err.println("Error while saving base model to file: " + filename);
            }

        }

    }
}
