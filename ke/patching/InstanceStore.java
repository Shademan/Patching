/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tud.ke.patching;

import java.util.Iterator;
import java.util.LinkedList;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Stores batches of instances stores n batches or all if you want to.
 *
 * @author SKauschke
 */
public class InstanceStore {

    int numBatches = Integer.MAX_VALUE;
    LinkedList<Instances> batches;

    public InstanceStore(int numBatches) {
        this.numBatches = numBatches;
        this.batches = new LinkedList<Instances>();
    }

    public InstanceStore() {
        this.batches = new LinkedList<Instances>();
    }

    public void addInstances(Instances inst) {

        batches.add(inst);
        
        while (batches.size() > this.numBatches) {
            batches.removeFirst(); // FIFO
        }
    }

    /**
     * Retrieves a batch of instances if exists, otherwise returns null.
     *
     * @param index
     * @return
     */
    public Instances getBatch(int index) {
        try {
            return batches.get(index);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println("Index " + index + " not found in InstanceStore.");
        }
        return null;
    }

    /**
     * Merges all the batches of instances and returns them.
     *
     * @return
     */
    public Instances getInstances() {
        return mergeAllInstances();
    }

    /**
     * Merges all the batches of instances.
     * Probably theres a way to speed this up?
     *
     * @param a
     * @param b
     * @return
     */
    private Instances mergeAllInstances() {

//        System.out.println("Merging instances of "+this.batches.size()+" batches.");
        if (this.batches.size() == 0) {
            return null;
        }

        Instances merged = new Instances(this.batches.getFirst());  // deep copy necessary!

        if (this.batches.size() == 1) {
            return merged;
        }

        for (int i = 1; i < this.batches.size(); i++) {
            Instances inst = this.batches.get(i);
            Iterator it = inst.iterator();
            while (it.hasNext()) {
                Instance in = (Instance) it.next();
                
                in.setWeight(i);
                merged.add(in);
            }
        }
        return merged;
    }
}
