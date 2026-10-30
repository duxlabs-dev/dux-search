self:
{
  config,
  lib,
  pkgs,
  ...
}:
with lib;
let
  tomlFormat = pkgs.formats.toml { };

  cfg = config.programs.dsearch;

  # Semantic search loads the system ONNX Runtime at run time. When the user
  # enables it, point the config at the Nix store copy unless they chose a path.
  semanticEnabled = cfg.config != null && (cfg.config.semantic.enabled or false);
  finalConfig =
    if semanticEnabled then
      recursiveUpdate {
        semantic.onnxruntime_lib = "${cfg.onnxruntimePackage}/lib/libonnxruntime.so";
      } cfg.config
    else
      cfg.config;
in
{
  options.programs.dsearch = {
    enable = mkEnableOption "Dux Search";
    package = mkPackageOption self.packages.${pkgs.stdenv.hostPlatform.system} "dsearch" { };

    onnxruntimePackage = mkOption {
      type = types.package;
      default = pkgs.onnxruntime;
      defaultText = literalExpression "pkgs.onnxruntime";
      description = ''
        ONNX Runtime (1.22 or newer) used by semantic search. Only referenced
        when `semantic.enabled = true` is set in `config`; the model files are
        downloaded at run time with `dsearch model download`.
      '';
    };

    config = mkOption {
      type = types.nullOr tomlFormat.type;
      default = null;
      description = ''
        dsearch configuration
      '';
    };
  };

  config = mkIf cfg.enable {
    home.packages = [ cfg.package ];

    systemd.user.services.dsearch = {
      Unit = {
        Description = "dsearch - Fast filesystem search service";
        Documentation = "https://github.com/duxlabs-dev/dux-search";
        After = [ "network.target" ];
      };

      Service = {
        Type = "simple";
        ExecStart = "${getExe cfg.package} serve";
        Restart = "on-failure";
        RestartSec = "5s";
        TimeoutStopSec = "30s";

        StandardOutput = "journal";
        StandardError = "journal";
        SyslogIdentifier = "dsearch";
      };

      Install = {
        WantedBy = [ "default.target" ];
      };
    };

    xdg.configFile."dsearch/config.toml" = mkIf (cfg.config != null) {
      source = tomlFormat.generate "dsearch.config.toml" finalConfig;
    };
  };
}
