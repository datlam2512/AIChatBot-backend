
from time import perf_counter

from app.services.chat_prompt_graph import chat_prompt_graph


# 1. Bộ câu hỏi kiểm thử
TEST_CASES = [
    {
        "question": "Python là gì?",
        "expected_mode": "general",
    },
    {
        "question": "Cảm ơn bạn!",
        "expected_mode": "general",
    },
    {
        "question": "Nova Lab có bao nhiêu ngày phép năm?",
        "expected_mode": "documents",
        "expected_keywords": ["14 ngày phép năm"],
    },
    {
        "question": "Nhân viên đi muộn quá 15 phút phải làm gì?",
        "expected_mode": "documents",
        "expected_keywords": ["15 phút", "10:00"],
    },
    {
        "question": "AI Bootcamp 2026 diễn ra ngày nào?",
        "expected_mode": "documents",
    },
    {
            "question": "Mất laptop của Nova Lab thì phải báo cho ai?",
            "expected_mode": "documents",
    },
    {
                "question": "Học phí AI Bootcamp 2026 là bao nhiêu?",
                "expected_mode": "documents",
    },
    {
                    "question": "Giải thích Gradient Descent",
                    "expected_mode": "general",
    },
    {
        "question": "Ở đâu?",
        "expected_mode": "documents",
        "expected_rewrite": "AI Bootcamp 2026",
        "history": [
            {
                "role": "user",
                "content": "AI Bootcamp 2026 diễn ra ngày nào?",
            },
            {
                "role": "assistant",
                "content": "Mình đã tìm thấy lịch sự kiện AI Bootcamp 2026.",
            },
        ],
    },
]


def contains_expected_keywords(chunks, keywords):
    """
    Kiểm tra có ít nhất một chunk chứa
    tất cả các từ khóa của đáp án không.
    """
    return any(
        all(
            keyword.casefold() in chunk.casefold()
            for keyword in keywords
        )
        for chunk in chunks
    )


def evaluate():
    total = len(TEST_CASES)
    correct_routes = 0

    retrieval_tests = 0
    retrieval_hits = 0

    rewrite_tests = 0
    rewrite_hits = 0

    total_latency = 0.0

    for index, case in enumerate(TEST_CASES, start=1):
        question = case["question"]

        messages = [
            *case.get("history", []),
            {"role": "user", "content": question},
        ]

        print("\n" + "=" * 60)
        print(f"TEST {index}: {question}")

        start = perf_counter()

        try:
            # 2. Chạy LangGraph thật
            result = chat_prompt_graph.invoke({
                "question": question,
                "messages": messages,
            })

            latency = perf_counter() - start
            total_latency += latency

            # 3. Kiểm tra Router
            actual_mode = result["mode"]
            expected_mode = case["expected_mode"]

            route_ok = actual_mode == expected_mode

            if route_ok:
                correct_routes += 1

            print("Expected mode:", expected_mode)
            print("Actual mode:", actual_mode)
            print("Routing:", "PASS" if route_ok else "FAIL")

            # 4. Kiểm tra Retrieval
            keywords = case.get("expected_keywords")

            if keywords:
                retrieval_tests += 1

                chunks = result.get("chunks", [])

                hit = contains_expected_keywords(
                    chunks,
                    keywords,
                )

                if hit:
                    retrieval_hits += 1

                print(
                    "Retrieval:",
                    "PASS" if hit else "FAIL",
                )

            # 5. Kiểm tra Question Rewriting
            expected_rewrite = case.get("expected_rewrite")

            if expected_rewrite:
                rewrite_tests += 1

                search_query = result.get(
                    "search_query", question
                )

                rewrite_ok = (
                    expected_rewrite.casefold()
                    in search_query.casefold()
                )

                if rewrite_ok:
                    rewrite_hits += 1

                print("Rewritten:", search_query)
                print(
                    "Rewrite:",
                    "PASS" if rewrite_ok else "FAIL",
                )

            print(f"Latency: {latency:.3f}s")

        except Exception as error:
            latency = perf_counter() - start
            total_latency += latency

            print("ERROR:", repr(error))
            print(f"Latency: {latency:.3f}s")

    # 6. Tổng kết kết quả
    routing_accuracy = correct_routes / total * 100

    hit_rate = (
        retrieval_hits / retrieval_tests * 100
        if retrieval_tests else 0
    )

    rewrite_rate = (
        rewrite_hits / rewrite_tests * 100
        if rewrite_tests else 0
    )

    avg_latency = total_latency / total

    print("\n" + "=" * 60)
    print("RAG EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Routing Accuracy: "
        f"{routing_accuracy:.1f}% "
        f"({correct_routes}/{total})"
    )

    print(
        f"Retrieval Hit@5: "
        f"{hit_rate:.1f}% "
        f"({retrieval_hits}/{retrieval_tests})"
    )

    print(
        f"Rewrite Check: "
        f"{rewrite_rate:.1f}% "
        f"({rewrite_hits}/{rewrite_tests})"
    )

    print(f"Average Graph Latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    evaluate()
