import sqlite3

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


def test_session_ids_and_titles_are_scoped_to_each_user(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    store.save_turn(
        user_id="user-a",
        session_id="shared-session",
        user_message="User A title",
        assistant_message="A answer",
    )
    store.save_turn(
        user_id="user-b",
        session_id="shared-session",
        user_message="User B title",
        assistant_message="B answer",
    )

    user_a_sessions = store.list_sessions(user_id="user-a")
    user_b_sessions = store.list_sessions(user_id="user-b")

    assert [(session["session_id"], session["title"]) for session in user_a_sessions] == [
        ("shared-session", "User A title")
    ]
    assert [(session["session_id"], session["title"]) for session in user_b_sessions] == [
        ("shared-session", "User B title")
    ]


def test_legacy_session_schema_is_migrated_without_data_loss(tmp_path):
    database = tmp_path / "legacy-chat.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
            VALUES ('shared-session', 'user-a', 'Existing title', '2026-01-01', '2026-01-01');
            """
        )

    store = ChatStore(str(database))
    store.save_turn(
        user_id="user-b",
        session_id="shared-session",
        user_message="New user title",
        assistant_message="New answer",
    )

    assert store.list_sessions(user_id="user-a")[0]["title"] == "Existing title"
    assert store.list_sessions(user_id="user-b")[0]["title"] == "New user title"


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

def test_chat_store_persists_structured_citations(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    citation = {
        "id": "S1",
        "source_name": "Guide.pdf",
        "page_number": 2,
        "web_view_link": "https://drive.google.com/file/d/file-1/view",
    }
    store.save_turn(
        user_id="user",
        session_id="session",
        user_message="Question",
        assistant_message="Grounded answer [S1]",
        assistant_citations=[citation],
    )

    messages = store.list_messages(user_id="user", session_id="session")

    assert messages[0]["citations"] == []
    assert messages[1]["citations"] == [citation]
