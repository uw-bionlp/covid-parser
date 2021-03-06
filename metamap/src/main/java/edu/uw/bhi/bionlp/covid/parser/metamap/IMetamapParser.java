package edu.uw.bhi.bionlp.covid.parser.metamap;

import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;

import java.util.HashSet;
import java.util.List;

/**
 *
 * @author wlau
 * @author ndobb
 */
public interface IMetamapParser {

    List<UMLSConcept> parseSentenceWithMetamap(String sentenceText, HashSet<String> semanticTypesOfInterest) throws Exception;
    
}