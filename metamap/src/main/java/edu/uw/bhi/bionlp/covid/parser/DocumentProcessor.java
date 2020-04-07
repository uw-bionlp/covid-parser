package edu.uw.bhi.bionlp.covid.parser;

import edu.uw.bhi.bionlp.covid.parser.grpcclient.AssertionClassifierClient;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetamapLiteParser;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapSentence;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.Sentence;

import java.util.List;

import org.javatuples.Pair;

import java.util.ArrayList;
import java.util.HashMap;

public class DocumentProcessor {

    MetamapLiteParser parser = new MetamapLiteParser();
    AssertionClassifierClient assertionClassifier = new AssertionClassifierClient();
    MetaMapConcept.Builder conceptBuilder = MetaMapConcept.newBuilder();

    public Pair<List<MetaMapSentence>, List<String>> processDocument(List<Sentence> sentences) {
        List<MetaMapSentence> mmSentences = new ArrayList<MetaMapSentence>();
        List<String> errors = new ArrayList<String>();
        HashMap<String,String> assertionCache = new HashMap<String,String>();

        /*
         * For each sentence.
         */
        for (int sId = 0; sId < sentences.size(); sId++) {
            List<MetaMapConcept> mmCons = new ArrayList<MetaMapConcept>();
            Sentence sentence = sentences.get(sId);

            /* 
             * Extract UMLS concepts.
             */
            List<UMLSConcept> concepts = new ArrayList<UMLSConcept>();
            try {
                concepts = parser.parseSentenceWithMetamap(sentence.getText());
            } catch (Exception ex) {
                String errorMsg = "Error: failed to parse with Metamap. Sentence: " + sId + " '" + sentence.getText() + "'. " + ex.getMessage();
                System.out.println(errorMsg);
                errors.add(errorMsg);
            }

            /* 
             * For each concept, if indexes valid, predict assertion.
             */
		    for (UMLSConcept concept : concepts) {
                String prediction = "present";
                try {

                    /*
                     * Check if this span has been seen before, if so use cached 
                     * rather than re-checking assertion.
                     */ 
                    String inputs = concept.getBeginCharIndex() + "|" + concept.getEndCharIndex();
                    if (assertionCache.containsKey(inputs)) {
                        prediction = assertionCache.get(inputs);

                    /*
                     * Else predict assertion.
                     */
                    } else {
                        prediction = assertionClassifier.predictAssertion(sentence.getText(), concept.getBeginCharIndex(), concept.getEndCharIndex());
                        assertionCache.put(inputs, prediction);
                    }
                } catch (Exception ex) {
                    String errorMsg = "Error: failed to predict concept. Sentence: " + sId + ", Concept: '" + concept.getConceptName() + "'. " + ex.getMessage();
                    System.out.println(errorMsg);
                    errors.add(errorMsg);

                /*
                 * Add the final concept.
                 */
                } finally {
                    MetaMapConcept mmCon = conceptBuilder
                        .setConceptName(concept.getConceptName())
                        .setCui(concept.getCUI())
                        .setBeginSentCharIndex(concept.getBeginCharIndex())
                        .setEndSentCharIndex(concept.getEndCharIndex())
                        .setBeginDocCharIndex(sentence.getBeginCharIndex() + concept.getBeginCharIndex())
                        .setEndDocCharIndex(sentence.getBeginCharIndex() + concept.getEndCharIndex())
                        .setSourcePhrase(concept.getPhrase())
                        .setPrediction(prediction)
                        .addAllSemanticTypes(concept.getSemanticTypeLabels())
                        .build();
                        mmCons.add(mmCon);
                }
            }
            MetaMapSentence mmSent = MetaMapSentence
                .newBuilder()
                .setBeginCharIndex(sentence.getBeginCharIndex())
                .setEndCharIndex(sentence.getEndCharIndex())
                .addAllConcepts(mmCons)
                .setId(sId)
                .setText(sentence.getText())
                .build();
            mmSentences.add(mmSent);
        }
        return new Pair<List<MetaMapSentence>, List<String>>(mmSentences, errors);
    }
}