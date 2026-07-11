"""Create or update a local Drive Agent user."""

import argparse
import getpass
import uuid

from services.auth import AuthService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("--role", choices=("admin", "user"), default="user")
    args = parser.parse_args()
    password = getpass.getpass("Password (minimum 8 characters): ")
    service = AuthService()
    service.create_user(str(uuid.uuid4()), args.username, password, args.role)
    print(f"User '{args.username}' created with role '{args.role}'.")


if __name__ == "__main__":
    main()
