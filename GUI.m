function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 16-Apr-2020 22:01:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

set(handles.axesOriginal, 'visible','off');
set(handles.axesResult1, 'visible','off');
set(handles.axesResult2, 'visible','off');
set(handles.axesResult3, 'visible','off');
set(handles.axesResult4, 'visible','off');
set(handles.axesResult5, 'visible','off');
addpath('others')

% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in buttonOpenFile.
function buttonOpenFile_Callback(hObject, eventdata, handles)
% hObject    handle to buttonOpenFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% 0.∂¡»°Õº∆¨≤¢œ‘ æ
[filename, pathname] = uigetfile( ...
{  '*.ppm;*.jpg;*.tif;*.png;*.gif','All Image Files';...
   '*.*','All Files' },'mytitle',...
   'data\stare-images\im0001.ppm');
im = imread([pathname, '\', filename]);
handles.im = im;
guidata(hObject,handles);
axes(handles.axesOriginal);
imshow(im,[]);
title('original');
method(im,handles);

function method(im,handles)
    cla(handles.axesResult1);
    cla(handles.axesResult2);
    cla(handles.axesResult3);
    cla(handles.axesResult4);
    cla(handles.axesResult5);
    t2=clock;
    axes(handles.axesResult1);imshow(matchedFilter(im,false));t1=t2;t2=clock;title(['matchedFilter:',num2str(etime(t2,t1)),'s']);
    axes(handles.axesResult2);imshow(gaussDerivativeFilter(im,false));t1=t2;t2=clock;title(['gaussDerivativeFilter:',num2str(etime(t2,t1)),'s']);
    axes(handles.axesResult3);imshow(laplacianPyramids(im,false));t1=t2;t2=clock;title(['laplacianPyramids:',num2str(etime(t2,t1)),'s']);
    axes(handles.axesResult4);imshow(principalCurvature(im,false));t1=t2;t2=clock;title(['principalCurvature:',num2str(etime(t2,t1)),'s']);
    axes(handles.axesResult5);imshow(PCAEnhance(im,false));t1=t2;t2=clock;title(['pcaEhance:',num2str(etime(t2,t1)),'s']);


% --- Executes on button press in pushbutton_matchedFilter.
function pushbutton_matchedFilter_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_matchedFilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    im = handles.im;
    matchedFilter(im,true);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    im = handles.im;
    gaussDerivativeFilter(im,true);
% --- Executes on button press in pushbutton_laplacianPyramids.
function pushbutton_laplacianPyramids_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_laplacianPyramids (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    im = handles.im;
    laplacianPyramids(im,true);
% --- Executes on button press in pushbutton_principalCurvature.
function pushbutton_principalCurvature_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_principalCurvature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    im = handles.im;
    principalCurvature(im,true);
% --- Executes on button press in pushbutton_pcaEnhance.
function pushbutton_pcaEnhance_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_pcaEnhance (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    im = handles.im;
    PCAEnhance(im,true);