def mm_response_to_dict(response):
    output = { 'id': response.id, 'text': response.text, 'sentences': [] }
    for sent in response.sentences:
        sentence = { 
            'id': sent.id, 
            'text': sent.text, 
            'concepts': [], 
            'beginCharIndex': sent.begin_char_index,
            'endCharIndex': sent.end_char_index,
        }
        for con in sent.concepts:
            concept = {
                'beginSentenceCharIndex': con.begin_sent_char_index,
                'endSentenceCharIndex': con.end_sent_char_index,
                'beginDocumentCharIndex': con.begin_doc_char_index,
                'endDocumentCharIndex': con.end_doc_char_index,
                'cui': con.cui,
                'semanticLabel': con.semantic_label,
                'sourcePhrase': con.source_phrase,
                'conceptName': con.concept_name,
                'prediction': con.prediction
            }
            sentence['concepts'].append(concept)
        output['sentences'].append(sentence)
    return output   