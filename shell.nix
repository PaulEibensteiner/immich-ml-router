{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      fastapi
      httpx
      uvicorn
      pytest
      pytest-asyncio
      respx
      anyio
      trio
    ]))
  ];
}