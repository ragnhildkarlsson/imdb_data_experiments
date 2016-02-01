from operator import itemgetter

import pickle_handler


GAVAGAI_COSINUS_SIMILARE_TERMS = "data/gavagai_pickles/gavagai_cosinus_similare_terms"


def get_default_keywords_dice(dice_ranked_keywords, weight_limit_reference_words):
    dice_ranked_keywords = dice_ranked_keywords[:100]
    reference_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words <= d[1]]
    context_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words > d[1]]
    return reference_words, context_words

def get_only_gavagai_cosinus_similare_keywords(gavagai_suggested_terms):    
    gavagai_cosinus_similare_terms = sorted(gavagai_suggested_terms, key=itemgetter('sumCosine'))    
    gavagai_cosinus_similare_terms = [term["term"] for term in gavagai_cosinus_similare_terms]
    gavagai_cosinus_similare_terms[:100]
    print(gavagai_cosinus_similare_terms)
    reference_words = gavagai_cosinus_similare_terms
    context_words = gavagai_cosinus_similare_terms
    return reference_words,context_words


