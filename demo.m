%% Load Data

data_shapes = {'armadillo', 'bunny', 'cube', 'dragon', 'pyramid', 'sphere'};

param_variants = {'_orth_pplastic_diffuse_0_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_10_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_20_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_30_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_40_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_50_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_60_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_70_deg_fov.mat', ...
                      '_persp_pplastic_diffuse_80_deg_fov.mat', ...
                  '_persp_pplastic_diffuse_90_deg_fov.mat'};

poolObj = gcp('nocreate');

if isempty(poolObj)
    % parpool('local', 8);
    parpool('Processes', 8);
end

parfor i = 1:length(data_shapes)
    current_shape = data_shapes{i};

    for j = 1:length(param_variants)
        current_param = param_variants{j};
        data_path = strcat('./data/', current_shape, current_param);
        % disp(class(data_path));
        % disp(strcat('Start processing: ', data_path));
        [py_normals_hfpol, py_normals_propagation, py_normals_lambertian] = compute_normals(data_path);
        parsave(strcat('./outputs/', current_shape, current_param), py_normals_hfpol, py_normals_propagation, py_normals_lambertian);
    end

end

delete(poolObj);

function parsave(fname, py_normals_hfpol, py_normals_propagation, py_normals_lambertian)
  save(fname, 'py_normals_hfpol', 'py_normals_propagation', 'py_normals_lambertian');
end

function [py_normals_hfpol, py_normals_propagation, py_normals_lambertian] = compute_normals(path_to_data)

    % load sampleData.mat
    % load ('./data/pyramid_persp_pplastic_diffuse_40_deg_fov.mat');
    load(path_to_data, "dolp", "aolp", "unpol", "spec", "mask");

    rho_est = transpose(dolp);
    phi_est = transpose(aolp);
    Iun_est = transpose(unpol);

    % clear images aolp dolp unpol;

    spec = transpose(spec);
    mask = transpose(mask);

    % Assume refractive index = 1.5 | 1.3 for Aluminium.
    n = 1.5;

    s = [0 0 10]';

    % % ? theta_est replaced with rho_est
    [s, ~, ~] = findLight(rho_est, phi_est, Iun_est, mask & ~spec, 3, s);

    % Compute angles, taking into account different model for specular pixels
    rho_est_combined = rho_diffuse(rho_est, n);
    rho_est_combined(spec) = rho_spec(rho_est(spec), n);
    phi_est_combined = phi_est;
    phi_est_combined(spec) = mod(phi_est(spec) + pi / 2, pi);

    %% LambertianSfP

    [~, lamb_height] = LambertianSFP(rho_est, phi_est, mask, n, ...
        s + [0.0001; 0; 0], ones(size(Iun_est)), Iun_est);

    [hx, hy, hz] = surfnorm(lamb_height);
    py_normals_lambertian = cat(3, hx, hy, hz);

    %% HfPol:

    % Compute boundary prior azimuth angles and weight
    [azi, Bdist] = boundaryPrior(mask);

    hfpol_height = HfPol(rho_est_combined, min(1, Iun_est), phi_est_combined, s, ...
        mask, false, spec, azi, Bdist);

    clear Iun_est phi_est_combined rho_est_combined s spec azi Bdist;

    [hx, hy, hz] = surfnorm(hfpol_height);
    py_normals_hfpol = cat(3, hx, hy, hz);

    %% Propagation:

    [~, propagation_height] = Propagation(rho_est, phi_est, mask, n);

    [hx, hy, hz] = surfnorm(propagation_height);
    py_normals_propagation = cat(3, hx, hy, hz);

end

% function [] = plot_height_data(height_data)
%     figure;
%     surf(height_data, 'EdgeColor', 'none', 'FaceColor', [0 0 1], 'FaceLighting', ...
%         'gouraud', 'AmbientStrength', 0, 'DiffuseStrength', 1); axis equal; light
% end
