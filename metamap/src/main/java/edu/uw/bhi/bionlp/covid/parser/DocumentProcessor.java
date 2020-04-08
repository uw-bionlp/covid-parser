package edu.uw.bhi.bionlp.covid.parser;

import java.util.List;
import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

import org.javatuples.Pair;
import edu.uw.bhi.bionlp.covid.parser.grpcclient.AssertionClassifierClient;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.MetaMapConcept;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.MetaMapSentence;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.Sentence;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetamapLiteParser;


public class DocumentProcessor {

    MetamapLiteParser parser = new MetamapLiteParser();
    AssertionClassifierClient assertionClassifier = new AssertionClassifierClient();
    MetaMapConcept.Builder conceptBuilder = MetaMapConcept.newBuilder();
    HashSet<String> allSemanticTypes = loadSemanticTypes();

    public Pair<List<MetaMapSentence>, List<String>> processDocument(List<Sentence> sentences, List<String> semanticTypesOfInterest) {
        /**
         * Initialize lists.
         */
        HashMap<String,String> assertionCache = new HashMap<String,String>();
        List<MetaMapSentence> mmSentences = new ArrayList<MetaMapSentence>();
        List<String> errors = new ArrayList<String>();
        HashSet<String> semanticTypes = semanticTypesOfInterest == null || semanticTypesOfInterest.size() == 0
            ? allSemanticTypes
            : collToHashSet(semanticTypesOfInterest);

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
                concepts = parser.parseSentenceWithMetamap(sentence.getText(), semanticTypes);
            } catch (Exception ex) {
                String errorMsg = "Error: failed to parse with Metamap. Sentence: " + sId + " '" + sentence.getText() + "'. " + ex.getMessage();
                System.out.println(errorMsg);
                errors.add(errorMsg);
            }

            /* 
             * For each concept, if indexes valid, predict assertion, defaulting to 'present'.
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
                        .clearSemanticTypes()
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

    HashSet<String> loadSemanticTypes() {
        HashSet<String> set = new HashSet<String>();
        try {
            String file = new String(Files.readAllBytes(new File("resources/umls/umls_semantic_type_labels.txt").toPath()));
            set = collToHashSet(Arrays.asList(file.split("\n")));
        } catch (Exception ex) {
            // Do nothing for now.
        }
        return set;
    }

    HashSet<String> collToHashSet(Collection<String> values) {
        HashSet<String> set = new HashSet<String>();
        for (String value : values) {
            set.add(value.trim());
        }
        return set;
    }
}