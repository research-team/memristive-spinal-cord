clear all % Очистка памяти
close all;
opengl('save','hardware');

[FileName,PathName,FilterIndex] = uigetfile({'*.txt','Data-files (*.txt)'}, ...
  'Input file with data for analyzer');
if FilterIndex==0
  return 
end

Col=8;
data=textread([PathName,FileName],'%s','delimiter','\t');
for i=1:6
    TitleS(i)=data(10+i);
end
data=data(25:end); %вырезаем лишнее и оставляем только числа
dataLength=length(data)/Col;
data=reshape(data,Col,dataLength); %конвертируем строку в матрицу нужного размера
rez=str2double(data);% переводим в нужный формат

CurrTitle = strcat(['File: ' FileName ]);
hf=figure('name',CurrTitle,'NumberTitle', 'off','MenuBar','none','NumberTitle', 'off');

%% Параметры
t=0.00025;
Tm=t*(dataLength-1);% Длина сигнала (с)
Fd=1/t;% Частота дискретизации (Гц)
FftL=4*4096;% Количество линий Фурье спектра
T=0:1/Fd:Tm;% Массив отсчетов времени

for i=1:4
Signal=rez(i+1,:);
%% Спектральное представление сигнала
FftS=abs(fft(Signal,FftL));% Амплитуды преобразования Фурье сигнала
FftS=(FftL/dataLength)*2*FftS./FftL;% Нормировка спектра по амплитуде
FftS(1)=FftS(1)/2;% Нормировка постоянной составляющей в спектре
%% Построение графиков
hS=subplot(4,2,2*i-1);% Выбор области окна для построения
plot(T,Signal,'color','r');% Построение сигнала
xlim([0,Tm]);
set(hS,'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', '-');
set(hS,'XMinorGrid','on','YMinorGrid','on','MinorGridLineStyle',':');
CurrTitle = strcat(['Signal ' TitleS{i}]);
title(CurrTitle);% Подпись графика
xlabel('Time (S)','FontName','Arial Cyr','FontSize',30);% Подпись оси х графика
ylabel('Amplitude (V)','FontName','Arial Cyr','FontSize',10);% Подпись оси у графика
hF=subplot(4,2,2*i);% Выбор области окна для построения
F=0:Fd/FftL:Fd/2-1/FftL;% Массив частот вычисляемого спектра Фурье
plot(F,FftS(1:length(F)));% Построение спектра Фурье сигнала
xlim([0,500]);
set(hF,'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', '-');
set(hF,'XMinorGrid','on','YMinorGrid','on','MinorGridLineStyle',':');
CurrTitle = strcat(['Spectrum ' TitleS{i}]);
title(CurrTitle);% Подпись графика
xlabel('Frequency (Hz)','FontName','Arial Cyr','FontSize',10);% Подпись оси х графика
ylabel('Amplitude','FontName','Arial Cyr','FontSize',10);% Подпись оси у графика
end;