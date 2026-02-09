from rich.traceback import install
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.orm import Session, sessionmaker
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from sqlite3 import Connection as SQLite3Connection

install(extra_lines=3)


# 定义数据库文件路径
ROOT_PATH = Path(__file__).parent.parent.parent.parent.absolute().resolve()
_DB_DIR = ROOT_PATH / "data"
_DB_FILE = _DB_DIR / "MaiBot.db"

# 确保数据库目录存在
_DB_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{_DB_FILE}"


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: "SQLite3Connection", connection_record):
    """
    为每个新的数据库连接设置 SQLite PRAGMA。

    这些设置优化了并发性能和数据安全性:
    - journal_mode=WAL: 启用预写式日志,提高并发性能
    - cache_size: 设置缓存大小为 64MB
    - foreign_keys: 启用外键约束
    - synchronous=NORMAL: 平衡性能和数据安全
    - busy_timeout: 设置1秒超时,避免锁定冲突
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 负值表示KB,64000KB = 64MB
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")  # NORMAL 模式在WAL下是安全的
    cursor.execute("PRAGMA busy_timeout=1000")  # 1秒超时
    cursor.close()


# 连接数据库
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@contextmanager
def get_db_session(auto_commit: bool = True) -> Generator[Session, None, None]:
    """
    获取数据库会话的上下文管理器 (推荐使用,自动提交)。

    Examples:
    ----
    .. code-block:: python
        # 方式1: 自动提交 (推荐 - 默认行为)
        with get_db_session() as session:
            user = User(name="张三", age=25)
            session.add(user)
            # 退出时自动 commit,无需手动调用

        # 方式2: 手动控制事务 (高级用法)
        with get_db_session(auto_commit=False) as session:
            user1 = User(name="张三", age=25)
            user2 = User(name="李四", age=30)
            session.add_all([user1, user2])
            session.commit()  # 手动提交

    Args:
        auto_commit (bool): 是否在退出上下文时自动提交（默认: True）。

    Yields:
        Session: SQLAlchemy 数据库会话

    注意:
        - 会话会在退出上下文时自动关闭
        - 如果发生异常，会自动回滚事务
        - auto_commit=True 时,成功执行完会自动提交
        - auto_commit=False 时,需要手动调用 session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        # 如果启用自动提交且没有异常,则提交事务
        if auto_commit:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session_manual():
    """获取数据库会话的上下文管理器 (手动提交模式)。"""
    return get_db_session(auto_commit=False)


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话的生成器函数。

    适用于依赖注入场景(如 FastAPI)。

    使用示例 (FastAPI):
    ----
    .. code-block:: python
        @app.get("/users/{user_id}")
        def read_user(user_id: int, db: Session = Depends(get_db)):
            return db.get(User, user_id)

    Yields:
        Session: SQLAlchemy 数据库会话
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class _AtomicContext:
    def __init__(self) -> None:
        self._session: Session | None = None

    def __enter__(self) -> Session:
        self._session = SessionLocal()
        self._session.begin()
        return self._session

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            self._session.close()


class DatabaseCompat:
    """兼容旧 db 调用接口（Peewee 风格），底层使用 SQLAlchemy。"""

    def connect(self, reuse_if_open: bool = True) -> None:
        # SQLAlchemy 由 engine 按需管理连接，这里保留兼容入口。
        _ = reuse_if_open

    def create_tables(self, models: list[type], safe: bool = True) -> None:
        _ = safe
        tables = [model.__table__ for model in models if hasattr(model, "__table__")]
        if not tables:
            return
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine, tables=tables)

    def atomic(self) -> _AtomicContext:
        return _AtomicContext()

    def execute_sql(self, sql: str):
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            conn.commit()
            return result

    def table_exists(self, model: type) -> bool:
        if not hasattr(model, "__tablename__"):
            return False
        inspector = sqlalchemy_inspect(engine)
        return inspector.has_table(model.__tablename__)


db = DatabaseCompat()
