from operator import itemgetter

import pickle_handler


GAVAGAI_COSINUS_SIMILARE_TERMS = "data/gavagai_pickles/gavagai_cosinus_similare_terms"


def get_default_keywords_dice(dice_ranked_keywords, weight_limit_reference_words):
    dice_ranked_keywords = dice_ranked_keywords[:100]
    reference_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words <= d[1]]
    context_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words > d[1]]
    return reference_words, context_words

# Get default dice keyword where reference word co-ocuring in bigram with category name is moved to context words
# return empty lists if no word where filtered
def get_dice_keyword_filter_1(category, bigram_delimeter, default_dice_reference_words, default_dice_context_words):
    all_dice_keywords = default_dice_context_words + default_dice_reference_words
    bigrams_with_category_name = [keyword for keyword in all_dice_keywords if bigram_delimeter in keyword and category in keyword]
    filtered_reference_words = set()
    for reference_word in default_dice_reference_words:
        for bigram_with_category_name in bigrams_with_category_name:
            if reference_word in bigram_with_category_name and not reference_word == category and not bigram_delimeter in reference_word and not category in reference_word and not reference_word in category:
                filtered_reference_words.add(reference_word)
    reference_words = [reference_word for reference_word in default_dice_reference_words if reference_word not in filtered_reference_words]
    filtered_reference_words = list(filtered_reference_words)
    context_words = default_dice_context_words + filtered_reference_words
    if not filtered_reference_words:
        print(category)
        print(filtered_reference_words)
        print(reference_words)
        reference_words =[]
        context_words = [] 
    return reference_words, context_words

    
def get_only_gavagai_paradigmatic_similare_keywords(gavagai_suggested_terms):
    gavagai_cosinus_similare_terms = sorted(gavagai_suggested_terms, key=itemgetter('sumCosine'))    
    gavagai_cosinus_similare_terms = [term["term"] for term in gavagai_cosinus_similare_terms]
    gavagai_cosinus_similare_terms[:100]
    print(gavagai_cosinus_similare_terms)
    reference_words = gavagai_cosinus_similare_terms
    context_words = gavagai_cosinus_similare_terms
    return reference_words,context_words


