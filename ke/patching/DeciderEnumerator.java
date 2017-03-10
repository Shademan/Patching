/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tud.ke.patching;

/**
 * A Decider is the element of a classifier that creates the final class decision.
 * In Rulesets, it is the rule that is triggered,
 * in decision trees, it is the leaf that makes the decision.
 * 
 * @author SKauschke
 */
public interface DeciderEnumerator {
 
    /**
     * Returns the total amount of deciders that exist (means: amount of rules, or
     * amount of leafs in the decision tree)
     * @return 
     */
    public int getAmountOfDeciders();
    
    /**
     * Returns the number of the decider that was responsible for the last 
     * instance that was classified
     * @return 
     */
    public int getLastUsedDecider();
    
    /**
     * Returns the number of the "default rule" which covers all previously
     * unclassified instances. 
     * In case of decision trees, i dont know what this should do :D
     * probably return -1 or so.
     * @return 
     */
    public int getDefaultDecider();
}
