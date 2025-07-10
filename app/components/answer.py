def answer_question(model, text: str) -> str:
    # print(text)
    model = model.configure(temperature=0.4)
    result = model.run(
        [
            {"role": "system", "text": "Строго следуй пользовательским инструкциям."},
            {
                "role": "user",
                "text": text,
            },
        ]
    )

    return result.alternatives[0].text

