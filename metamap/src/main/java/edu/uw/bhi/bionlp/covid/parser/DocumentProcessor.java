package edu.uw.bhi.bionlp.covid.parser;

import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierOuterClass.Sentence;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifierclient.AssertionClassifierClient;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetamapLiteParser;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapSentence;

import java.util.List;
import java.util.ArrayList;

public class DocumentProcessor {

    MetamapLiteParser parser = new MetamapLiteParser();
    AssertionClassifierClient assertionClassifier = new AssertionClassifierClient();
    MetaMapSentence.Builder sentenceBuilder = MetaMapSentence.newBuilder();
    MetaMapConcept.Builder conceptBuilder = MetaMapConcept.newBuilder();
    int ngramSize = 5;

    public List<MetaMapSentence> processDocument(String note) {
        List<MetaMapSentence> mmSentences = new ArrayList<MetaMapSentence>();
        
        /* 
         * Chunk into sentences.
         */
        List<Sentence> sentences = new ArrayList<Sentence>();
        try {
            sentences = assertionClassifier.detectSentences(note);
        } catch (Exception ex) {
            System.out.println("Error: failed to chunk sentences. " + ex.getMessage());
        }

        /*
         * For each sentence.
         */
        for (int sId = 0; sId < sentences.size(); sId++) {
            Sentence sentence = sentences.get(sId);
            sentenceBuilder
                .setBeginCharIndex(sentence.getBeginCharIndex())
                .setEndCharIndex(sentence.getEndCharIndex())
                .setText(sentence.getText());

            /* 
             * Extract UMLS concepts.
             */
            List<UMLSConcept> concepts = new ArrayList<UMLSConcept>();
            try {
                concepts = parser.parseSentenceWithMetamap(sentence.getText());
            } catch (Exception ex) {
                System.out.println("Error: failed to parse with Metamap. Sentence: '" + sentence.getText() + "'. " + ex.getMessage());
            }

            /* 
             * For each concept, if indexes valid, predict assertion.
             */
		    for (UMLSConcept concept : concepts) {
                String ngram = concept.getNgram(sentence.getText(), ngramSize);
                String prediction = "present";
                try {
                    prediction = assertionClassifier.predictAssertion(ngram, concept.getBeginTokenIndex(), concept.getEndTokenIndex());
                } catch (Exception ex) {
                    System.out.println("Error: failed to predict concept. Concept: '" + concept.getConceptName() + "'. " + ex.getMessage());
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
                        .setSemanticLabel(concept.getSemanticTypeLabels())
                        .build();
                    sentenceBuilder.addConcepts(mmCon);
                }
            }
            mmSentences.add(sentenceBuilder.build());
        }
        return mmSentences;
    }
}