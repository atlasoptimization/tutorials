function [Distance_Matrix]=Make_Distance_Matrix(Point_list_1,Point_list_2)


% [Distance_Matrix]=Make_Distance_Matrix(Point_List_1,Point_List_2)
% This function takes as Input two Matrices containing Point Coordinates. 
% The output of this function is a Distance_Matrix containing at position
% (i,j) the distance between Point i in Point_List 1 and Point j in 
% Point_List 2.
% The input Lists should have the form: 
%         PL 1                                    PL 2
% [X_s1   X_s2  X_s3 ... X_sn_est]          [X_p1   X_p2  X_p3 ... X_pn_data]
% [Y_s1   Y_s2  Y_s3 ... Y_sn_est]          [Y_p1   Y_p2  Y_p3 ... Y_pn_data]
% [...    ...    ... ...  ...    ]          [...    ...    ... ...      ... ]
%
% In the case of 2 dimensional Kriging these Lists are just
%         PL 1                                    PL 2
% [X_s1   X_s2  X_s3 ... X_sn_est]          [X_1   X_2  X_3 ... X_n_data]
% [Y_s1   Y_s2  Y_s3 ... Y_sn_est]          [Y_1   Y_2  Y_3 ... Y_n_data]
%
% The distance between every point s_i (to be estimated) with coordinates 
% (X_si,Y_si) to every data point p_i (used for estimation) is calculated.

% Look at explanations and formulas for derivation. 
% stp= Coordinates of estimation locations transposed*Coordinates of data
% locations.

% Of course this approach is not limited to only two coordinates; in
% particular one coordinate yields the (unsquared) difference between every
% pair of points.

ptp=Point_list_1'*Point_list_1;
sts=Point_list_2'*Point_list_2;
stp=Point_list_1'*Point_list_2;
d_squared= bsxfun(@plus,diag(ptp),diag(sts)')-2*stp;
Distance_Matrix=sqrt(d_squared);

end