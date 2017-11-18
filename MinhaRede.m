% puxa o arquivo de excel, onde est�o os dados
dataset = xlsread ('DadosSecagem.xlsx','Planilha1');
% escolhe as colunas que fazem parte do dataset
dataset = dataset(2:end,1:5);


% Nomeia cada coluna da database com suas respectivas vari�veis
% Normaliza as vari�veis para valores entre 0 e 1

tempo = dataset(:,1);
t=tempo./max(tempo);

temperatura = dataset(:,2);
temp=temperatura./max(temperatura);

peso = dataset(:,3);
peso=peso./max(peso);

parte = dataset (:,4);
parte = parte./max(parte);

umidade = dataset (:,5);

% Cria a transposta do dataset, para ficar no formato certo
dataset_transp = [t,temp,peso,parte,umidade]';

% ESCOLHA DO VETOR DE VALIDA��O 
% vetor [colunas a serem retiradas para valida��o]
vetor_val = [1,10,50];

%------------------------------------------------------------------------
%----------------- PARAMETROS VARI�VEIS ---------------------------------
%-------------------------------------------------------------------------

HiddenSize = 5;       % num. de neuronios na camada intermedi�ria
meta_v = 0.997;       % Valor de R^2 desejado na valida��o
meta_t = meta_v;       % Valor de R^2 desejado no treino
treino = 75/100;      % fra��o de separa��o para treino
validacao = 15/100;    % fra��o de separa��o para valida��o
teste = 10/100;        % fra��o de separa��o para teste
epochs = 5000;         % num. maximo de epocas
max_iteracoes = 1000;  % num. maximo de itera��es
erro_permitido = 1.e-12;  % erro permitido

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

%retira dos dados originais as colunas para valida��o externa
dataset_t = dataset_transp;
dataset_t = dataset_t(:,setdiff(1:size(dataset_t,2),vetor_val));

%cria matriz de entradas e saidas da valida��o
entradav = dataset_transp(1:end-1,vetor_val);
saidav = dataset_transp(end,vetor_val);

%cria matriz de entradas e saidas usadas para o treino da rede
entradat = dataset_t(1:end-1,:);
saidat = dataset_t(end,:);

% configura��es da Rede Neural 
net = newff(entradat,saidat,HiddenSize); % newff, newcf, newelm
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
net.divideParam.trainRatio = treino;
net.divideParam.valRatio = validacao;
net.divideParam.testRatio = teste;
net.trainFcn =  'trainbr';  % Levenberg-Marquardt
net.performFcn = 'mse';  % Mean squared error  (mse, sse, sae)
net.trainParam.epochs= epochs;
net.trainParam.showWindow=0;
net.trainParam.goal=erro_permitido;
sinal = 1;
i = 0;

% Cria vari�veis que ser�o utilizadas no treino
s=[0;0];
maior_r=0.7;

%%%%%% TREINO DA REDE 
%enquanto o sinal for diferente de -2, fa�a:
while sinal~=-2 
    
   
    [net,tr] = train(net,entradat,saidat); %comando do matlab para treinar (convergir uma rede)
    % depois de treinada, a rede estima as saidas, tanto do treino como da valida��o
    saidat_rede = sim(net,entradat); %Estimations
    saidav_rede = sim(net,entradav);
    % Calcula o erro entre a real sa�da e a sa�da estimada da rede
    errot = abs(saidat-saidat_rede)./saidat;
    errov = abs(saidav-saidav_rede)./saidav;
    % Calcula o R^2 do treino e valida��o
    R2t = corrcoef(saidat_rede,saidat);
    R2v = corrcoef(saidav_rede,saidav);
    % Gera variavel sinal
    % OBS.: Se R^2 for maior que o desejado, sinal ser� negativo
    sinal1 = meta_v - abs(R2v(1,2));
    sinal2 = meta_t - abs(R2t(1,2));
    % Arredonda e soma os valores da vari�vel sinal.
    % Se os dois sinais foram negativos (desejado) para de treinar
    sinal = sum(sign([sinal1;sinal2]));
    
    % mostra na janela de comando a itera��o que est� e os R^2 equivalentes
    i
    r = [R2t(1,2), R2v(1,2)]
    
    % --------------------------------------------------------------------
    % ---- GRAVA OS MELHORES RESULTADOS ENCONTRADOS NO TREINO ------------
    %---------------------------------------------------------------------
    
    % A partir da segunda itera��o, se o menor dos R^2 entre o treino e
    % valida��o superar o maior R^2 encontrado at� ent�o:
    
    if i > 2 && min(r) > maior_r;
       
        % nomeia um novo valor para maior R^2 encontrado
        maior_r = min(r);
        maiores_r = r;
     
     % plota um grafico de correla��o para o treino
     figure(1)
        postreg(saidat,saidat_rede);
        xlabel('Umidade relativa experimental','FontSize',14) % nome do eixo x
        ylabel('Umidade relativa estimada','FontSize',14) % nome do eixo y
        title( ['Sa�das x Objetivos R�=' num2str(R2t(1,2))] ) % titulo do grafico
 
   % Plota figuras de ajuste dos pontos retirados para valida��o 
    %figure(2)
    %[mv,av,rv] = postreg(saidav,saidav_rede);
    
    
    % Grava os pesos encontrados
    m_pesos_imp = net.IW{1}; % Pesos da hidden layer para output
    m_pesos_hl = net.LW{2}; % Pesos da hidden layer para output
    m_bias_imp = net.b{1}; % Vieses imput 
    m_bias_out = net.b{2}; % Vieses outputs
    
    % Grava a rede encontrada
    melhor_net = net;
    
    % Pausa para poder ver os graficos ao longo das itera��es
    pause(0.0001);
    
    end 
    
 %--------------------------------------------------------------------
 %--------------------------------------------------------------------
    
    i = i+1; 
    
    %Gera uma matriz que adiciona os valores de R^2 na ultima linha apos
    %cada itera��o
     s = [s,r'];
            
    % Se n�o convergir antes do maximo de itera��es estipuladas, for�a a
    %parada do treino
    if i== max_iteracoes
        sinal = -2;
        i
    end
   
end

% mostra a matriz dos R^2 apresentados durante o treino
 result = s(:,2:end)'   






