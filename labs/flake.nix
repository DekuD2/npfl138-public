{
  description = "Deep Learning (npfl138).";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      templates.default = {
        path = ./flake-template;
        description = "Deep Learning (npfl138) flake template.";
        welcomeText = ''
          # npfl138 flake template
          ## Important
          Make sure the flake is placed in the `/labs` directory of the npfl138 repository (https://github.com/ufal/npfl138). The directory should also contain `pyproject.toml`.
          ## Usage
          Type `nix develop` to enter the dev shell with the required dependencies.
        '';
      };
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python311
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH =
              let
                libpackages = pkgs.pythonManylinuxPackages.manylinux1 ++ [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ];
              in
                (lib.makeLibraryPath libpackages) + ":/run/opengl-driver/lib";
            };

            # Increase timeout because nvidia packages are very large.
            UV_HTTP_TIMOEOUT = 5000;

            shellHook = ''
              unset PYTHONPATH
              uv sync
              # source .venv/bin/activate
              # If you use nushell, replace the line above with:
              exec nu -e "overlay use .venv/bin/activate.nu"
            '';
          };
        }
      );
    };
}
