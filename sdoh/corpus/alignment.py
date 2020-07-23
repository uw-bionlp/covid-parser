import re


def align(text, tokens, start=0):
    '''
    Find indices of tokens within text
    
    args:
        text = document as string
        tokens = document as list of list of strings 
           (list of sentences, where each sentence is a list of strings)
    '''

    # Create copy of text
    #text_tmp = text
    text_tmp = text[start:]
    
    # Character offset
    #offset = 0
    offset = start
    
    # Initialize document indices
    doc_indices = []
    
    # Iterate over tokenized sentences within document
    for sent in tokens:
                       
        # Sentence pattern, ignoring white space
        sent_pat = "\s*".join([re.escape(tok) for tok in sent])
        
        # Find current sentence
        sent_m = re.search(sent_pat, text_tmp, re.MULTILINE)

        # Text not found
        assert bool(sent_m), '''
        Sent = {}\n\n{}\n
        Sent pat = {}\n\n{}\n
        <Text =>\n{}\n\n{}\n'''.format( \
             " ".join(sent), '='*72, sent_pat, '='*72, text_tmp, '='*72)
        
        # Start of current sentence
        sent_start = sent_m.start()

        # Delete text preceding start a sentence
        text_tmp = text_tmp[sent_start:]

        # Update offset
        offset += sent_start
       
        # Initialize sentence indices
        sent_indices = []
        
        # Iterate over tokens and sentence
        for tok in sent:

            # Search for next token at start of string
            tok_m = re.search(re.escape(tok), text_tmp)
                        
            # Tax not found
            assert bool(tok_m), '''
            Token = {}\n\n{}\n
            <Text =>\n{}\n\n{}\n
            <Text, full =>\n{}:'''.format( \
                                    tok, '='*72, text_tmp, '='*72, text)
            
            # Get start, end, and offset
            start_tok = tok_m.start()
            end_tok = tok_m.end()

            # Check extracted text
            assert tok == tok_m.group(0), \
            '''Token = {}\n
              Text found = {}\n
              Start = {}\nStop = {}\n\n
              Text = {}'''.format(tok, tok_m.group(0), start_tok, end_tok, text_tmp)
            
            # Append token indices to sentence
            indices = (start_tok + offset, end_tok + offset)
            sent_indices.append(indices)

            # Check extracted text
            text_found = text[indices[0]:indices[1]]
            assert tok == text_found, \
            '''Token = {}\n
               Text found = {}\n
               Indices = {}\n
               Text = {}'''.format(tok, text_found, indices, text)
            
            # Delete found characters from text
            text_tmp = text_tmp[end_tok:]

            # Update offset
            offset += end_tok
        
        # Append sentence to document    
        doc_indices.append(sent_indices)

    # Check indices
    assert len(doc_indices) == len(tokens), \
    '''Sentence count:
    \tFrom indices = {}
    \tFrom tokens  = {}
    Indices = {}
    Tokens = {}'''.format(len(doc_indices), len(tokens), \
                                                    doc_indices, tokens)
    
    for di, tok in zip(doc_indices, tokens):
        assert len(di) == len(tok), \
        '''Token count:
        From indices = {}
        From tokens  = {}
        Indices = {}
        Tokens = {}'''.format(len(di), len(to), di, tok)            

    return doc_indices
