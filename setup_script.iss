; Inno Setup Script for AI Human Fall Detection System
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Human Fall Detection System"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "wqminhh"
#define MyAppExeName "FallDetectionApp.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".myp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; Use a new GUID for each released version if you want Windows to treat it as a new installation record.
AppId={{A41B9A70-25E5-4F70-A26A-7C849D4B6209}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\FallDetectionApp
UninstallDisplayIcon={app}\{#MyAppExeName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output setup file settings
OutputDir=dist
OutputBaseFilename=FallDetectionApp_Setup_v{#MyAppVersion}
SetupIconFile=app_icon.ico
SolidCompression=yes
Compression=lzma2/ultra64
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Source directory from PyInstaller --onedir build output
Source: "dist\FallDetectionApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"; Tasks: desktopicon

[Run]
Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Filename: "{app}\{#MyAppExeName}"; Flags: nowait postinstall skipifsilent
