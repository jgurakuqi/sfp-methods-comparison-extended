function [N, height] = Propagation(rho, phi, mask, n)
    %PROPAGATION Shape-from-polarisation by boundary propagation
    %   Inputs:
    %      rho    - rows by cols matrix of DOP values
    %      phi    - rows by cols matrix of phase angles
    %      mask   - rows by cols binary foreground mask
    %      n      - refractive index
    %
    %   Outputs:
    %      N      - rows by cols by 3 matrix containing surface normals
    %      height - height map obtained by integrating N using lsqintegration
    %
    % This is essentially a re-implementation of the technique used in:
    %
    % Miyazaki, D., Tan, R.T., Hara, K. and Ikeuchi, K. Polarization-based
    % inverse rendering from a single view. In Proc. ICCV, pp. 982-987, 2003.
    %
    % and
    %
    % G. A. Atkinson and E. R. Hancock. Recovery of Surface Orientation from
    % Diffuse Polarization. IEEE Transactions on Image Processing 15:1653-1664,
    % June 2006.
    %
    % It disambiguates polarisation normals beginning at the boundary. Boundary
    % normals are disambiguated to minimise angular error to the outward facing
    % boundary normal. Normals are then propagated into the interior in
    % descending order of zenith angle, with the disambiguation being chosen
    % that maximises smoothness with respect to already-disambiguated normals.
    %
    % William Smith
    % 2016

    % Invert degree of diffuse polarisation expression to compute zenith angle
    temp = ((2 .* rho + 2 .* n .^ 2 .* rho - 2 .* n .^ 2 + n .^ 4 + rho .^ 2 + 4 .* n .^ 2 * rho .^ 2 - n .^ 4 .* rho .^ 2 - 4 .* n .^ 3 .* rho .* (- (rho - 1) .* (rho + 1)) .^ (1/2) + 1) ./ (n .^ 4 .* rho .^ 2 + 2 .* n .^ 4 .* rho + n .^ 4 + 6 .* n .^ 2 .* rho .^ 2 + 4 .* n .^ 2 .* rho - 2 .* n .^ 2 + rho .^ 2 + 2 .* rho + 1)) .^ (1/2);
    temp = min(real(temp), 1);
    theta = acos(temp);

    % Pad twice to avoid boundary problems (a 7x7 neighbourhood around pixels
    % is considered for smoothness)
    mask = pad(pad(mask));
    theta = pad(pad(theta));
    phi = pad(pad(phi));
    [rows, cols] = size(mask);

    N = zeros(rows, cols, 3) .* NaN;

    B = bwboundaries(mask, 8, 'noholes');
    boundary = mask ~= mask;
    % This keeps track of which pixels have been disambiguated:
    available_estimates = mask ~= mask;

    % Fill in boundary normals with disambiguation that is closest to outward
    % facing normal
    B{1}(end + 1, :) = B{1}(1, :);

    for i = 1:size(B{1}, 1) - 1
        r = B{1}(i, 1);
        c = B{1}(i, 2);
        available_estimates(r, c) = true;
        boundary(r, c) = true;
        azi = atan2(B{1}(i + 1, 2) - B{1}(i, 2), B{1}(i + 1, 1) - B{1}(i, 1));
        n1 = [sin(phi(r, c)) * sin(theta(r, c)); cos(phi(r, c)) * sin(theta(r, c)); cos(theta(r, c))];
        n2 = [sin(phi(r, c) + pi) * sin(theta(r, c)); cos(phi(r, c) + pi) * sin(theta(r, c)); cos(theta(r, c))];
        nb = [cos(azi) * sin(theta(r, c)); sin(azi) * sin(theta(r, c)); cos(theta(r, c))];

        if dot(n1, nb) > dot(n2, nb)
            N(r, c, :) = n1;
        else
            N(r, c, :) = n2;
        end

    end

    interior = mask & ~boundary;

    % Sort zenith angles of interior pixels into descending order
    [~, idx] = sort(theta(interior), 1, 'descend');
    [col, row] = meshgrid(1:cols, 1:rows);

    r_interior = row(interior);
    c_interior = col(interior);

    clear row col interior boundary B temp;

    list1 = -3:3;
    list2 = -3:3;

    % Generate all combinations using meshgrid
    [A, B] = meshgrid(list1, list2);
    neigh_indexes = [A(:), B(:)];
    to_remove = 25;
    neigh_indexes(to_remove, :) = [];
    % disp(neigh_indexes);

    % At each iteration, we choose the unprocessed pixel with smallest zenith
    % angle that has at least one neighbour

    neighbourhood = zeros(49, 2);

    while ~isempty(idx)
        flag = false;
        selected = 1;

        % Consider pixels in ranked-theta order to find first one with at least
        % one neighbour (over 7x7 neighbourhood)
        while flag == false
            % neighbourhood = [];
            index = 1;

            r_interior_selected = r_interior(idx(selected));
            c_interior_selected = c_interior(idx(selected));

            for c = 1:49 - 1
                i = neigh_indexes(c, 1);
                j = neigh_indexes(c, 2);
                r_selected = r_interior_selected + i;
                c_selected = c_interior_selected + j;

                if available_estimates(r_selected, c_selected)
                    neighbourhood(index, :) = [r_selected c_selected];
                    index = index + 1;
                end

            end

            % for i = -3:3

            %     for j = -3:3

            %         if (i ~= 0) || (j ~= 0)
            %             r_selected = r_interior_selected + i;
            %             c_selected = c_interior_selected + j;

            %             if available_estimates(r_selected, c_selected)
            %                 neighbourhood(index, :) = [r_selected c_selected];
            %                 index = index + 1;
            %             end

            %         end

            %     end

            % end

            % if ~isempty(neighbourhood) % We need at least one neighbour to test smoothness
            if index > 1
                flag = true;
            else
                selected = selected + 1;
            end

        end

        % We now have a pixel with at least one neighbour
        r = r_interior(idx(selected));
        c = c_interior(idx(selected));

        Ns = zeros(index - 1, 3);

        for i = 1:index - 1
            neigh1 = neighbourhood(i, 1);
            neigh2 = neighbourhood(i, 2);
            Ns(i, :) = [N(neigh1, neigh2, 1) N(neigh1, neigh2, 2) N(neigh1, neigh2, 3)];
        end

        % Compute which local solution has smaller mean angular deviation from
        % its neighbours, our definition of smoothness
        phi_rc = phi(r, c);
        theta_rc = theta(r, c);
        n1 = [sin(phi_rc) * sin(theta_rc); cos(phi_rc) * sin(theta_rc); cos(theta_rc)];
        n2 = [sin(phi_rc + pi) * sin(theta_rc); cos(phi_rc + pi) * sin(theta_rc); cos(theta_rc)];

        if mean(acos(Ns * n1)) < mean(acos(Ns * n2))
            N(r, c, :) = n1;
        else
            N(r, c, :) = n2;
        end

        available_estimates(r, c) = true;
        idx(selected) = [];
    end

    % Unpad estimated normals and mask
    N = N(3:end - 2, 3:end - 2, :);
    N = real(N);
    mask = mask(3:end - 2, 3:end - 2);
    % Integrate normals into height map
    height = lsqintegration(N, mask, false);

end
