
def get_default_keywords_dice(dice_ranked_keywords, weight_limit_reference_words):
    dice_ranked_keywords = dice_ranked_keywords[:100]
    reference_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words <= d[1]]
    context_words = [d[0] for d in dice_ranked_keywords if weight_limit_reference_words > d[1]]
    return reference_words, context_words


def get_dice_keywords_filter_word_appear_in_bigram_with_category_name(category_name,
                                                                      default_reference_words,
                                                                      default_context_words,
                                                                      bigram_delimeter):    
    reference_words = default_reference_words
    context_words = default_context_words
    bigrams_with_category_name = set([r for r in reference_words if bigram_delimeter in r and category_name in r])
    reference_words_in_bigram_with_category_name = set([r for r in reference_words if not bigram_delimeter in r and not r == category_name and not r in bigrams_with_category_name])
    reference_words = [r for r in reference_words if r not in reference_words_in_bigram_with_category_name]
    for r in reference_words_in_bigram_with_category_name:
        context_words.add(r)
    # TODO REMOVE
    if reference_words_in_bigram_with_category_name:
        print(category_name)
        print(reference_words_in_bigram_with_category_name)
    return reference_words, context_words


