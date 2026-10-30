{
  description = "Indexed filesystem search in GO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      goModVersion =
        let
          content = builtins.readFile ./go.mod;
          lines = builtins.filter builtins.isString (builtins.split "\n" content);
          goLines = builtins.filter (l: builtins.match "go [0-9]+\\..*" l != null) lines;
          matched =
            if goLines != [ ] then builtins.match "go ([0-9]+)\\.([0-9]+).*" (builtins.head goLines) else null;
        in
        if matched != null then
          {
            major = builtins.elemAt matched 0;
            minor = builtins.elemAt matched 1;
          }
        else
          {
            major = "1";
            minor = "25";
          };
      goForPkgs = pkgs: pkgs.${"go_${goModVersion.major}_${goModVersion.minor}"};

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = (f:
        nixpkgs.lib.genAttrs supportedSystems (system:
          f nixpkgs.legacyPackages.${system} system
        )
      );

    in
    {
      packages = forAllSystems (
        pkgs: system:
        let
          inherit (pkgs) lib;
          dsearchVersion = "2.0.0";
        in
        {
          dsearch = (pkgs.buildGoModule.override { go = goForPkgs pkgs; }) {
            pname = "dsearch";
            version = dsearchVersion;
            
            src = ./.;
            vendorHash = "sha256-MkRBJPsHHgH2z/hTt8Z6SM/+KNQ8PuipBALW8qbJldE=";

            # dsearch-embed is the model helper that dsearch starts next to itself
            subPackages = [
              "cmd/dsearch"
              "cmd/dsearch-embed"
            ];

            ldflags = [
              "-s"
              "-w"
              "-X main.Version=${dsearchVersion}"
            ];

            meta = {
              description = "Indexed filesystem search in GO";
              homepage = "https://github.com/duxlabs-dev/dux-search";
              mainProgram = "dsearch";
              license = lib.licenses.mit;
              platforms = lib.platforms.unix;
            };
          };

          default = self.packages.${system}.dsearch;
        }
      );

      homeModules = {
        default = self.homeModules.dsearch;
        dsearch = import ./distro/nix self;
      };
    };
}
