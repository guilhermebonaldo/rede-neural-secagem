% puxa o arquivo de excel, onde estão os dados
dataset = xlsread ('DadosSecagem.xlsx','Planilha1');
% escolhe as colunas que fazem parte do dataset
dataset = dataset(2:end,1:5);


% Nomeia cada coluna da database com suas respectivas variáveis
% Normaliza as variáveis para valores entre 0 e 1

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

% ESCOLHA DO VETOR DE VALIDAÇÃO 
% vetor [colunas a serem retiradas para validação]
vetor_val = [1,10,50];

%------------------------------------------------------------------------
%----------------- PARAMETROS VARIÁVEIS ---------------------------------
%-------------------------------------------------------------------------

HiddenSize = 5;       % num. de neuronios na camada intermediária
meta_v = 0.997;       % Valor de R^2 desejado na validação
meta_t = meta_v;       % Valor de R^2 desejado no treino
treino = 75/100;      % fração de separação para treino
validacao = 15/100;    % fração de separação para validação
teste = 10/100;        % fração de separação para teste
epochs = 5000;         % num. maximo de epocas
max_iteracoes = 1000;  % num. maximo de iterações
erro_permitido = 1.e-12;  % erro permitido

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

%retira dos dados originais as colunas para validação externa
dataset_t = dataset_transp;
dataset_t = dataset_t(:,setdiff(1:size(dataset_t,2),vetor_val));

%cria matriz de entradas e saidas da validação
entradav = dataset_transp(1:end-1,vetor_val);
saidav = dataset_transp(end,vetor_val);

%cria matriz de entradas e saidas usadas para o treino da rede
entradat = dataset_t(1:end-1,:);
saidat = dataset_t(end,:);

% configurações da Rede Neural 
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

% Cria variáveis que serão utilizadas no treino
s=[0;0];
maior_r=0.7;

%%%%%% TREINO DA REDE 
%enquanto o sinal for diferente de -2, faça:
while sinal~=-2 
    
   
    [net,tr] = train(net,entradat,saidat); %comando do matlab para treinar (convergir uma rede)
    % depois de treinada, a rede estima as saidas, tanto do treino como da validação
    saidat_rede = sim(net,entradat); %Estimations
    saidav_rede = sim(net,entradav);
    % Calcula o erro entre a real saída e a saída estimada da rede
    errot = abs(saidat-saidat_rede)./saidat;
    errov = abs(saidav-saidav_rede)./saidav;
    % Calcula o R^2 do treino e validação
    R2t = corrcoef(saidat_rede,saidat);
    R2v = corrcoef(saidav_rede,saidav);
    % Gera variavel sinal
    % OBS.: Se R^2 for maior que o desejado, sinal será negativo
    sinal1 = meta_v - abs(R2v(1,2));
    sinal2 = meta_t - abs(R2t(1,2));
    % Arredonda e soma os valores da variável sinal.
    % Se os dois sinais foram negativos (desejado) para de treinar
    sinal = sum(sign([sinal1;sinal2]));
    
    % mostra na janela de comando a iteração que está e os R^2 equivalentes
    i
    r = [R2t(1,2), R2v(1,2)]
    
    % --------------------------------------------------------------------
    % ---- GRAVA OS MELHORES RESULTADOS ENCONTRADOS NO TREINO ------------
    %---------------------------------------------------------------------
    
    % A partir da segunda iteração, se o menor dos R^2 entre o treino e
    % validação superar o maior R^2 encontrado até então:
    
    if i > 2 && min(r) > maior_r;
       
        % nomeia um novo valor para maior R^2 encontrado
        maior_r = min(r);
        maiores_r = r;
     
     % plota um grafico de correlação para o treino
     figure(1)
        postreg(saidat,saidat_rede);
        xlabel('Umidade relativa experimental','FontSize',14) % nome do eixo x
        ylabel('Umidade relativa estimada','FontSize',14) % nome do eixo y
        title( ['Saídas x Objetivos R²=' num2str(R2t(1,2))] ) % titulo do grafico
 
   % Plota figuras de ajuste dos pontos retirados para validação 
    %figure(2)
    %[mv,av,rv] = postreg(saidav,saidav_rede);
    
    
    % Grava os pesos encontrados
    m_pesos_imp = net.IW{1}; % Pesos da hidden layer para output
    m_pesos_hl = net.LW{2}; % Pesos da hidden layer para output
    m_bias_imp = net.b{1}; % Vieses imput 
    m_bias_out = net.b{2}; % Vieses outputs
    
    % Grava a rede encontrada
    melhor_net = net;
    
    % Pausa para poder ver os graficos ao longo das iterações
    pause(0.0001);
    
    end 
    
 %--------------------------------------------------------------------
 %--------------------------------------------------------------------
    
    i = i+1; 
    
    %Gera uma matriz que adiciona os valores de R^2 na ultima linha apos
    %cada iteração
     s = [s,r'];
            
    % Se não convergir antes do maximo de iterações estipuladas, força a
    %parada do treino
    if i== max_iteracoes
        sinal = -2;
        i
    end
   
end

% mostra a matriz dos R^2 apresentados durante o treino
 result = s(:,2:end)'   






