def decoded_answer_list_to_string(answer, pad_token: str = "∅"):
    # if string - convert to list:
    if isinstance(answer, str):
        answer = list(answer)
    # collapse each sequence of chars (length up to 3) that are siblings and duplicates into one
    for i in range(len(answer) - 1, 0, -1):
        if answer[i] == answer[i - 1]:
            answer.pop(i)
    answer = "".join(answer).replace(pad_token, "")
    return answer
