function [q]=InverseKinematics(R,o)
 
conv=180/pi;
parameter=0.4;
oc=o-(parameter * R(:,3));
 
%joint angle theta 1
if(isequal(oc(2),0.0123))
    theta11 = pi/2;
else
    theta11=atan2(oc(2),oc(1));
end

theta12=theta11+pi;
 
%joint angle theta 3
D=[(oc(1).^2)+(oc(2).^2)+((oc(3)-0.4).^2)-(0.16)-(0.16)]/(2*0.16);
theta3u=atan2(D,sqrt(abs(1-(D^2))));
theta3d=atan2(D,-sqrt(1-D.^2));
 
%joint angle theta 2
k=((oc(3)-0.4)^2+ oc(2)^2 +oc(1)^2)/(2*sqrt((oc(3)-0.4)^2 + oc(2)^2 +oc(1)^2)*0.4);
theta21=atan2((oc(3)-0.4),sqrt(oc(1).^2 + oc(2).^2))-atan2(sqrt(1-k^2),k);
theta22=atan2((oc(3)-0.4),sqrt(oc(1).^2 + oc(2).^2))-atan2(-sqrt(1-k^2),k);
theta23=atan2((oc(3)-0.4),-sqrt(oc(1).^2 + oc(2).^2))-atan2(sqrt(1-k^2),k);
theta24=atan2((oc(3)-0.4),-sqrt(oc(1).^2 + oc(2).^2))-atan2(-sqrt(1-k^2),k);
 
%sol 1
dh_mat=[pi/2 0 0.4 theta11; 0 0.4 0 theta22;pi/2 0 0 theta3u];
R3_0= eye(3);
R3_0= R3_0 * FT(dh_mat);
R6_3=R3_0'*R;
theta51=atan2(sqrt(1-(R6_3(3,3))^2),R6_3(3,3));
theta41=atan2(R6_3(2,3),R6_3(1,3));
theta61=atan2(R6_3(3,2),-R6_3(3,1));
 
 
ans1=conv*[theta11 theta22 theta3u theta41 theta51 theta61];
disp(ans1);
 
%sol 2
dh_mat=[0 pi/2 0.4 theta11;0.4 0 0 theta22;0 pi/2 0 theta3u];
R3_0= eye(3);
R3_0= R3_0 * FT(dh_mat);
R6_3=R3_0'*R;
theta52=atan2(-sqrt(1-(R6_3(3,3))^2),R6_3(3,3));
theta42=atan2(-R6_3(2,3),-R6_3(1,3));
theta62=atan2(-R6_3(3,2),R6_3(3,1));
 
 
ans2=conv*[theta11 theta22 theta3u theta42 theta52 theta62];
disp(ans2);
 
%sol 3
 
dh_mat=[0 pi/2 0.4 theta11;0.4 0 0 theta21;0 pi/2 0 theta3d];
R3_0= eye(3);
R3_0= R3_0 * FT(dh_mat);
R6_3=R3_0'*R;
theta51=atan2(sqrt(1-(R6_3(3,3))^2),R6_3(3,3));
theta41=atan2(R6_3(2,3),R6_3(1,3));
theta61=atan2(R6_3(3,2),-R6_3(3,1));
 
 
ans3=conv*[theta11 theta21 theta3d theta41 theta51 theta61];
disp(ans3);
 
%sol 4
 
dh_mat=[0 pi/2 0.4 theta11;0.4 0 0 theta21;0 pi/2 0 theta3d];
R3_0= eye(3);
R3_0= R3_0 * FT(dh_mat);
R6_3=R3_0'*R;
theta52=atan2(-sqrt(1-(R6_3(3,3))^2),R6_3(3,3));
theta42=atan2(-R6_3(2,3),-R6_3(1,3));
theta62=atan2(-R6_3(3,2),R6_3(3,1));
 
ans4=conv*[theta11 theta21 theta3d theta42 theta52 theta62];
disp(ans4);
 

 
q=[ans1;ans2;ans3;ans4;ans5;ans6;ans7;ans8];
disp(q);
 
end