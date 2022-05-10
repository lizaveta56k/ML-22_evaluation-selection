import nox
from nox.sessions import Session

locations = "poetry_task_9", "tests"

@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    session.install("joblib")
    session.install("numpy")
    session.install("sklearn")
    session.install("pytest")
    session.install("click")
    session.install("mlflow")

    session.run("pytest", *args)


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.install("mypy")
    session.install("joblib")
    session.install("numpy")
    session.install("sklearn")
    session.install("pytest")
    session.install("click")
    session.install("mlflow")
    
    session.run("mypy", *args)
