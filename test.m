clear; clc; close all

% Wczytanie danych
data = readtable('crabs.dat');

% Ekstrakcja cech liczbowych (FL, RW, CL, CW, BD)
features = table2array(data(:, 4:8));

% Kodowanie płci: M = 0, F = 1
sex_str = data.sex;
sex_num = strcmp(sex_str, 'F');

% Dodajemy płeć jako 6. cechę
features = [features, sex_num];
features_unnormalized = features;

% Zapisujemy min i max do późniejszej normalizacji nowych danych
min_vals = min(features(:,1:5));
max_vals = max(features(:,1:5));

% Normalizacja cech liczbowych (pierwsze 5) – bez płci
features(:,1:5) = (features(:,1:5) - min_vals) ./ (max_vals - min_vals);

% Zakodowanie gatunku: B = 0, O = 1
species_str = data.sp;
species_num = strcmp(species_str, 'O');

% Usuwanie odstających danych na podstawie cech 1–5
Q1 = quantile(features(:,1:5), 0.25);
Q3 = quantile(features(:,1:5), 0.75);
IQR = Q3 - Q1;
is_outlier = any(features(:,1:5) < Q1 - 1.5 * IQR | features(:,1:5) > Q3 + 1.5 * IQR, 2);

features_clean = features(~is_outlier, :);
species_clean = species_num(~is_outlier);

features_unnormalized_clean = features_unnormalized(~is_outlier, :);

% Podział na zbiór treningowy i testowy
n = size(features_clean, 1);
idx = randperm(n);
n_train = round(0.8 * n);

train_data = features_clean(idx(1:n_train), :);
train_labels = species_clean(idx(1:n_train));

test_data = features_clean(idx(n_train+1:end), :);
test_labels = species_clean(idx(n_train+1:end));

test_data_unnormalized = features_unnormalized_clean(idx(n_train+1:end), :);

% Parametry sieci
input_dim = 6;
output_dim = 1;
initial_eta = 0.5;
l_iter = 3000;

% Inicjalizacja wag
%rng(56);
weights = rand(output_dim, input_dim);
errors = zeros(l_iter, 1);

% Uczenie sieci WTA
for iter = 1:l_iter
    eta = initial_eta * (1 - iter / l_iter);
    total_error = 0;

    for i = 1:size(train_data, 1)
        x = train_data(i, :)';
        distances = vecnorm(weights - x', 2, 2);
        [min_dist, winner_idx] = min(distances);

        weights(winner_idx, :) = weights(winner_idx, :) + eta * (x' - weights(winner_idx, :));
        weights(winner_idx, :) = weights(winner_idx, :) / norm(weights(winner_idx, :));

        total_error = total_error + min_dist;
    end
    errors(iter) = total_error / size(train_data, 1);
end

% Wykres błędu
figure;
plot(1:l_iter, errors, 'b', 'LineWidth', 1.5);
xlabel('Iteracja'); ylabel('Średni błąd'); title('Uczenie sieci');
grid on;

% Przypisanie neuronów do gatunków
neuron_species = zeros(output_dim, 1);
for i = 1:output_dim
    assigned = [];
    for j = 1:size(train_data,1)
        x = train_data(j, :)';
        distances = vecnorm(weights - x', 2, 2);
        [~, winner_idx] = min(distances);
        if winner_idx == i
            assigned(end+1) = train_labels(j);
        end
    end
    if ~isempty(assigned)
        neuron_species(i) = mode(assigned);
    else
        neuron_species(i) = -1;
    end
end

% Testowanie przypisań
assignments = zeros(size(test_data,1),1);
predicted_species = zeros(size(test_data,1),1);

for i = 1:size(test_data,1)
    x = test_data(i, :)';
    distances = vecnorm(weights - x', 2, 2);
    [~, winner_idx] = min(distances);
    assignments(i) = winner_idx;
    predicted_species(i) = neuron_species(winner_idx);
end

% Wykres przypisań
counts = histcounts(assignments, 0.5:1:12.5);
figure;
bar(1:12, counts);
xlabel('Neuron wyjściowy');
ylabel('Liczba przypisań');
title('Przypisania danych testowych do neuronów');
grid on;

% Dokładność klasyfikacji
valid_idx = predicted_species ~= -1;
accuracy = sum(predicted_species(valid_idx) == test_labels(valid_idx)) / sum(valid_idx);
fprintf('Dokładność klasyfikacji: %.2f%%\n', accuracy * 100);

true_labels = test_labels(valid_idx);
preds = predicted_species(valid_idx);

% Obliczanie TP, TN, FP, FN
TP = sum((true_labels == 1) & (preds == 1));
TN = sum((true_labels == 0) & (preds == 0));
FP = sum((true_labels == 0) & (preds == 1));
FN = sum((true_labels == 1) & (preds == 0));

% Czułość i specyficzność
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

fprintf('Czułość: %.2f%%\n', sensitivity * 100);
fprintf('Specyficzność : %.2f%%\n', specificity * 100);

%  Podpisanie poszczególnych neuronów
for i = 1:output_dim
    if neuron_species(i) ~= -1
        fprintf('Neuron %d reprezentuje klasę %d\n', i, neuron_species(i));
    end
end
%% --- PRZEWIDYWANIE GATUNKU DLA WIELU KRABÓW --- %%

% Wprowadź dane kilku krabów jako wiersze:
% Każdy wiersz: [FL, RW, CL, CW, BD, Sex]
% Płeć: M = 0, F = 1
new_crabs = [
    16.4, 13.0, 35.7, 41.8, 15.2, 0; %Rodzaj B, płeć M
    10.1,  9.3, 20.9, 24.4,  8.4, 1; %Rodzaj B, płeć F
    20.6, 14.4, 42.8, 46.5, 19.6, 0; %Rodzaj O, płeć M
    19.7, 16.7, 39.9, 43.6, 18.2, 1  %Rodzaj O, płeć F
];

% Normalizacja (cechy 1–5)
new_crabs(:,1:5) = (new_crabs(:,1:5) - min_vals) ./ (max_vals - min_vals);

% Przewidywanie gatunku
fprintf('\n--- Przewidywania dla nowych krabów ---\n');
for i = 1:size(new_crabs,1)
    x = new_crabs(i,:)';
    distances = vecnorm(weights - x', 2, 2);
    [~, winner_idx] = min(distances);
    predicted_species = neuron_species(winner_idx);

    fprintf('Krab %d: ', i);
    if predicted_species == 0
        fprintf('Gatunek B\n');
    elseif predicted_species == 1
        fprintf('Gatunek O\n');
    else
        fprintf('Nieznany gatunek (nieprzypisany neuron)\n');
    end
end
