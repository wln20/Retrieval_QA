query_templates = {
    'en': """
        ### Instruction: Use your knowledge and the supporting facts (if provided) to determine the answer of the question correctly.\
        Only output the the letter corresponding to your answer choice, which should be chosen from A, B, C and D. \
        Do not output anything other than your choice. \

        ### Facts: {}

        ### Question: {}
        
        ### Choices:
        A. {}
        B. {}
        C. {}
        D. {}

        ### Answer: 
    """,

    'zh_cn': """
        ### 任务：利用你的知识和给定的事实（如果提供）来正确确定问题的答案。\
        只输出与你的答案选择相对应的字母，该字母应从 A、B、C 和 D 中选择。 \
        除了你的选择之外，不要输出任何其他内容。 \

        ### 事实：{}

        ### 问题： {}
        
        ### 选项：
        A. {}
        B. {}
        C. {}
        D. {}

        ### 回答：
    """,   

    'zh_tw': """
        ### 任務：利用你的知識和給定的事實（如果提供）來正確確定問題的答案。\
        只輸出與你的答案選擇相對應的字母，該字母應從 A、B、C 和 D 中選擇。 \
        除了你的選擇之外，不要輸出任何其他內容。 \
        
        ### 事實：{}

        ### 問題： {}
        
        ### 選項：
        A. {}
        B. {}
        C. {}
        D. {}

        ### 回答：
    """,

    'ja': """
        ### タスク：知識と裏付けとなる事実 (提供されている場合) を使用して、質問の答えを正しく判断します。\
        回答の選択肢に対応する文字のみを出力します。A、B、C、D から選択する必要があります。 \
        選択した内容以外は出力しないでください。 \

        ### 事実：{}

        ### 質問：{}
        
        ### 選択肢：
        A. {}
        B. {}
        C. {}
        D. {}

        ### 答え：
    """,

    'es': """
        ### Instrucción: Utilice su conocimiento y los datos de respaldo (si se proporcionan) para determinar la respuesta correcta a la pregunta.\
        Solo envíe la letra correspondiente a su elección de respuesta, que debe elegirse entre A, B, C y D. \
        No genere nada que no sea su elección. \

        ### Hechos: {}

        ### Pregunta: {}
        
        ### Opciones:
        A. {}
        B. {}
        C. {}
        D. {}

        ### Respuesta:
    """,

    'de': """
        ### Anleitung: Nutzen Sie Ihr Wissen und die unterstützenden Fakten (falls vorhanden), um die Antwort auf die Frage richtig zu bestimmen.\
        Geben Sie nur den Buchstaben aus, der Ihrer Antwortauswahl entspricht und aus A, B, C und D ausgewählt werden sollte. \
        Geben Sie nichts anderes als Ihre Auswahl aus. \

        ### Fakten: {}

        ### Frage: {}
        
        ### Auswahlmöglichkeiten:
        A. {}
        B. {}
        C. {}
        D. {}

        ### Antwort:   
    """,

    'ru': """
        ### Инструкция: используйте свои знания и подтверждающие факты (если они есть), чтобы правильно определить ответ на вопрос.\
        Выведите только букву, соответствующую вашему варианту ответа, который следует выбрать из A, B, C и D. \
        He выводите ничего, кроме вашего выбора. \

        ### Факты: {}

        ### Вопрос: {}
        
        ### Варианты:
        A. {}
        B. {}
        C. {}
        D. {}

        ### Отвечать:
    """
}


def format_query_prompt(example, lang='en', mode='without_doc', retrieved_docs=[]):   
    """
    `mode`: choose from 'without_doc', 'with_correct_doc', 'with_retrieved_docs'
    if mode == 'with_retrieved_docs', then `retrieved_docs` should be a list of retrieved docs.
    """
    if mode == 'without_doc':
        doc = 'None'
    elif mode == 'with_correct_doc':
        doc = '\n' + example['doc']
    elif mode == 'with_retrieved_docs':
        docs = []
        for j in range(len(retrieved_docs)):
            docs[j] = f"{j}. {retrieved_docs[j]}"  # 1. doc1
        doc = '\n' + '\n'.join(docs)

    query = query_templates[lang].format(
        doc,
        example['query'],
        example['choice'][0][1],    # A's content
        example['choice'][1][1],    # B's content
        example['choice'][2][1],    # C's content
        example['choice'][3][1],    # D's content
    )
    ans = example["answer"]

    return query, ans