import argparse
import os
import sys

import requests


ALLOWED_MODELS = ("resnet18", "efficientnet_b0")


class TritonAdminError(RuntimeError):
    pass


class TritonAdmin:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str) -> requests.Response:
        try:
            response = requests.request(method, f"{self.base_url}{path}", timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            raise TritonAdminError(str(error)) from error

    def index(self) -> list[dict]:
        return self.request("POST", "/v2/repository/index").json()

    def change(self, model_name: str, action: str) -> None:
        if model_name not in ALLOWED_MODELS:
            raise TritonAdminError(f"Modelo no permitido: {model_name}")
        self.request("POST", f"/v2/repository/models/{model_name}/{action}")

    def switch(self, model_name: str) -> None:
        if model_name not in ALLOWED_MODELS:
            raise TritonAdminError(f"Modelo no permitido: {model_name}")
        for current in self.index():
            if current.get("name") in ALLOWED_MODELS and current.get("state") == "READY":
                if current["name"] != model_name:
                    self.change(current["name"], "unload")
        self.change(model_name, "load")
        ready = [item["name"] for item in self.index() if item.get("state") == "READY"]
        if ready != [model_name]:
            raise TritonAdminError(f"Estado inesperado después del cambio: {ready}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Administra modelos explícitos de Triton.")
    parser.add_argument("action", choices=("status", "load", "unload", "switch"))
    parser.add_argument("model", nargs="?", choices=ALLOWED_MODELS)
    parser.add_argument("--url", default=os.getenv("TRITON_URL", "http://localhost:8000"))
    args = parser.parse_args()
    if args.action != "status" and args.model is None:
        parser.error("load, unload y switch requieren un modelo")
    try:
        admin = TritonAdmin(args.url)
        if args.action == "status":
            for item in admin.index():
                print(f"{item.get('name')}: {item.get('state')}")
        elif args.action == "switch":
            admin.switch(args.model)
            print(f"Modelo activo: {args.model}")
        else:
            admin.change(args.model, args.action)
            print(f"{args.action}: {args.model}")
        return 0
    except TritonAdminError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())