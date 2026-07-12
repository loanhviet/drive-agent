from services.chat import ChatStore


def test_chat_store_orders_messages_and_isolates_users(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    store.save_turn(
        user_id="user-a",
        session_id="session-1",
        user_message="First question",
        assistant_message="First answer",
    )
    store.save_turn(
        user_id="user-a",
        session_id="session-1",
        user_message="Second question",
        assistant_message="Second answer",
    )
    store.save_turn(
        user_id="user-b",
        session_id="session-1",
        user_message="Private question",
        assistant_message="Private answer",
    )

    messages = store.list_messages(user_id="user-a", session_id="session-1")
    assert [(message["role"], message["content"]) for message in messages] == [
        ("user", "First question"),
        ("assistant", "First answer"),
        ("user", "Second question"),
        ("assistant", "Second answer"),
    ]
    assert store.list_messages(user_id="user-b", session_id="session-1")[0]["content"] == "Private question"


def test_chat_store_context_limit_keeps_latest_messages(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    store.save_turn(
        user_id="user", session_id="session", user_message="old", assistant_message="old answer"
    )
    store.save_turn(
        user_id="user", session_id="session", user_message="new", assistant_message="new answer"
    )

    assert [message["content"] for message in store.list_messages(
        user_id="user", session_id="session", limit=2
    )] == ["new", "new answer"]


def test_chat_store_creates_lists_and_deletes_sessions(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    session = store.create_session(user_id="user-a")
    store.save_turn(
        user_id="user-a",
        session_id=session["session_id"],
        user_message="A useful conversation title that should be shown",
        assistant_message="answer",
    )
    other_session = store.create_session(user_id="user-b")

    sessions = store.list_sessions(user_id="user-a")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session["session_id"]
    assert sessions[0]["title"].startswith("A useful conversation")
    assert store.delete_session(user_id="user-a", session_id=other_session["session_id"]) is False
    assert store.delete_session(user_id="user-a", session_id=session["session_id"]) is True
    assert store.list_messages(user_id="user-a", session_id=session["session_id"]) == []
