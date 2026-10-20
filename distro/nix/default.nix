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
in
{
  options.programs.dsearch = {
    enable = mkEnableOption "Dux Search";
    package = mkPackageOption self.packages.${pkgs.stdenv.hostPlatform.system} "dsearch" { };

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
      source = tomlFormat.generate "dsearch.config.toml" cfg.config;
    };
  };
}
