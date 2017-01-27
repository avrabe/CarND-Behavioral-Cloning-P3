#include <File.au3>
#include <GUIConstantsEx.au3>
#include <ColorConstants.au3>
#include <EditConstants.au3>
#include <WindowsConstants.au3>

global $gitPath = @HomePath & "\git\CarND-Project-3\"
global $simulatorPath = @HomePath&"\Downloads\simulator-windows-64\Default Windows desktop 64-bit.exe"
global $anacondaPath = @HomePath&"\AppData\Local\Continuum\Anaconda3\Scripts\activate.bat"


func allstop()
ProcessClose("Python.exe")
ProcessClose("Python.exe")
ProcessClose("Python.exe")
ProcessClose("Python.exe")
ProcessClose("Python.exe")
WinClose ( "Self Driving Car Nanodegree Program  Configuration")
WinClose ( "Self Driving Car Nanodegree Program")
WinClose ( "Self Driving Car Nanodegree Program  Configuration")
WinClose ( "drive")
EndFunc

Func Example($model)
    ; Create a GUI with various controls.
    Local $hFileOpen = FileOpen($gitPath&"\foo.log", $FO_APPEND)

    Local $hGUI = GUICreate("Acceptor", 250,150 ,0 ,0)
	GUICtrlCreateLabel ("Running " & $model, 10,20)
    Local $idFAIL = GUICtrlCreateButton("Fail", 10,40)
    Local $idPASS = GUICtrlCreateButton("Pass", 50,40)
    Local $idABORT = GUICtrlCreateButton("Abort",10,70)
    Local $idMyedit = GUICtrlCreateEdit("", 10, 100, 200, 40, $ES_AUTOVSCROLL + $WS_VSCROLL)

    GUICtrlSetBkColor($idFAIL, $COLOR_RED)
    GUICtrlSetBkColor($idPASS, $COLOR_GREEN)

    MouseMove(28,90)

    ; Display the GUI.title DriveGetDrive
    GUISetState(@SW_SHOW, $hGUI)

    ; Loop until the user exits.
    While 1
        Switch GUIGetMsg()
		Case $GUI_EVENT_CLOSE, $idFAIL
		        FileWriteLine($hFileOpen, $model &" fail: " & GUICtrlRead ( $idMyedit) & @CRLF)
                ExitLoop
		Case $GUI_EVENT_CLOSE, $idPASS
		        FileWriteLine($hFileOpen, $model &" pass: " & GUICtrlRead ( $idMyedit) &@CRLF)
                ExitLoop
	    Case $GUI_EVENT_CLOSE, $idABORT
			 FileWriteLine($hFileOpen, $model &" aborted: " & GUICtrlRead ( $idMyedit) & @CRLF)
			 FileWriteLine($hFileOpen, "----------------------------" & @CRLF)
			 FileClose($hFileOpen)
			 allstop()
			 Exit
        EndSwitch
    WEnd
    FileClose($hFileOpen)
    ; Delete the previous GUI and all controls.
    GUIDelete($hGUI)
EndFunc   ;==>Example

Func close_simulator()
WinClose ( "Self Driving Car Nanodegree Program  Configuration")
WinClose ( "Self Driving Car Nanodegree Program")
EndFunc

Func simulator()
close_simulator()

Run($simulatorPath)
WinWaitActive("[CLASS:#32770]")
#ControlClick("Self Driving Car Nanodegree Program  Configuration","", "[CLASS:Button; INSTANCE:1]")
ControlClick("Self Driving Car Nanodegree Program  Configuration","", "[CLASS:Button; INSTANCE:3]")
WinWaitActive("[CLASS:UnityWndClass]")
WinMove ( "Self Driving Car Nanodegree Program","", 0,0)
Sleep(2500)
MouseClick ( "" , 519, 509 )
EndFunc

FUNC drive($model)
RUN("cmd" & " /k " & $anacondaPath & " carnd-term1")
Local $hWnd = WinWaitActive("[CLASS:ConsoleWindowClass]")
WinMove ( $hWnd,"", 1027,44)
SendKeepActive ($hWnd)
Sleep(1000)
Send ("title drive{ENTER}")
Sleep(500)
Send ("PATH=c:\cuda\bin;%PATH%{ENTER}")
Sleep(500)
Send ("set CUDA_VISIBLE_DEVICES=1{ENTER}")
Sleep(500)
Send ("cd %HOMEPATH%/git/CarND-Project-3{ENTER}")
Sleep(500)
Send ("python drive.py "&$model&"{ENTER}")
SendKeepActive ("")
simulator()

Example($model)
ProcessClose("Python.exe")
sleep(1000)
WinClose($hWnd)
close_simulator()
EndFunc

allstop()

Local $aFileList = _FileListToArray ( $gitPath, "*.json", $FLTA_FILES )
Local $first = False
For $model In $aFileList
   if not $first then
	  $first = True
   Else
	  Call("drive", $model)
   endif
Next

