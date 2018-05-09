#include <Servo.h>
int a1 = 0,a2 = -3,a3 = 3,a4 = 0;
int arr[8][8][4]={{{128,79,91,160},{118,87,97,167},{109,96,103,167},{94,102,111,165},{77,105,111,165},{63,99,108,160},{51,94,99,167},{42,86,95,163}},
{{122,71,81,155},{116,79,86,159},{104,79,87,163},{93,83,92,164},{80,87,96,159},{68,89,97,156},{57,84,92,155},{48,80,85,153}},
{{118,67,73,150},{110,66,70,158},{101,68,72,162},{89,72,73,165},{79,73,73,166},{70,73,73,164},{61,72,70,163},{53,64,62,165}},
{{115,52,51,160},{107,57,59,161},{100,59,59,163},{91,65,67,157},{82,64,64,158},{73,61,62,158},{64,60,62,156},{57,50,48,165}},
{{112,44,45,148},{106,49,50,147},{98,52,52,154},{88,52,50,159},{82,54,50,160},{74,53,50,158},{66,53,52,152},{60,41,35,155}},
{{109,31,23,148},{102,32,25,151},{96,34,25,157},{87,33,23,156},{82,33,23,160},{76,33,23,157},{68,33,23,156},{62,31,18,155}},
{{108,21,12,137},{101,25,16,141},{95,23,12,143},{87,23,14,143},{82,23,12,147},{75,23,12,147},{69,23,12,145},{63,23,12,139}},
{{105,15,12,114},{100,17,17,111},{94,20,20,117},{88,21,25,118},{82,24,25,118},{76,25,25,115},{72,21,25,102},{66,23,25,102}}};

for(int i=0;i<8;i++){
  for(int j=0;j<8;j++){
    arr[i][j][0]+=a1;
    arr[i][j][1]+=a2;
    arr[i][j][2]+=a3;
    arr[i][j][3]+=a4;
  }
}
int index;
Servo s1,s2,s3,s4,s5;
int dir,ang,counter=0,a,b, i =0;
bool aCh = false,bCh = false;
int incomingByte = 0;
int ind1 = 0;
String pos[4] = {" "," "," "," "};
int num1 = 0, num2 = 0,num3 = 0,num4 = 0;
int ind3 = 0;
void initPos(){
  if(s2.read()<120){
    dir = 1;
  }
  else{
    dir = -1;
  }
  ang = s2.read();
  while(ang!=120){
    ang+=dir;
    s2.write(ang);
    delay(15);
  }
  delay(2000);
  if(s3.read() < 45){
    dir = 1;
  }
  else{
    dir = -1;
  }
  ang = s3.read();
  while(ang!=45){
    ang+=dir;
    s3.write(ang);
    delay(15);
  }
  //s3.write(45);
  s1.write(90);
  delay(1000);
  //s4.write(170);
  if(s4.read() < 160){
    dir = 1;
  }
  else{
    dir = -1;
  }
  ang = s4.read();
  while(ang!=160){
    ang+=dir;
    s4.write(ang);
    delay(15);
  }
  delay(700);
}
void setup() {
  s1.attach(2);
  s2.attach(3);
  s3.attach(4);
  s4.attach(5);
  s5.attach(6);
  s5.write(70);
  index = 0;
  dir = 0;
  ind3 = 32;
  Serial.begin(9600);
  initPos();
}
void movePiece(int index,int mode){
  if(mode==1){
   for(int ang=s5.read();ang>75;ang--){
      s5.write(ang);
      delay(15);
    }
  }
  initPos();
  int row = index/8;
  int col = index%8;
  ang = s4.read();
  if(arr[row][col][3]>ang){
    dir = 1;
  }
  else{
    dir = -1;
  }
  while(ang!=arr[row][col][3]){
    /*Serial.print(4);
    Serial.print(" ");
    Serial.println(ang);*/
    ang+=dir;
    s4.write(ang);
    delay(15);
  }
  delay(750);
  if(arr[row][col][0] > 90){
    dir = 1;
  }
  else{
    dir = -1;
  }
  ang = s1.read();
  while(ang != arr[row][col][0]){
    /*Serial.print(1);
    Serial.print(" ");
    Serial.println(ang);*/
    ang+=dir;
    s1.write(ang);
    delay(15);
  }
  delay(750);
  ang = s3.read();
  if(arr[row][col][2]>45){
    dir = 1;
  }
  else{
    dir = -1;
  }
  while(ang!=arr[row][col][2]){
    /*Serial.print(3);
    Serial.print(" ");
    Serial.println(ang);*/
    ang+=dir;
    s3.write(ang);
    delay(15);
  }
  //s3.write(arr[index][2]);
  delay(750);
  //s1.write(arr[index][0]);
  if(arr[row][col][1]>120){
    dir = 1;
  }
  else{
    dir = -1;
  }
  ang = s2.read();
  while(ang!=arr[row][col][1]){
    /*Serial.print(2);
    Serial.print(" ");
    Serial.println(ang);*/
    ang+=dir;
    s2.write(ang);
    delay(15);
  }
  //s2.write(arr[index][1]);
  //s4.write(arr[index][3]);
  delay(2000);  
  if(mode==-1){
    s4.write(s4.read()+10);
    //s3.write(s3.read()+20);
    s2.write(s2.read()-5);
    //s4.write(s4.read()+5);
    for(int ang=s5.read();ang>70;ang--){
      s5.write(ang);
      delay(15);
    }
  }
  else{
    for(int ang=s5.read();ang<90;ang++){
      s5.write(ang);
      delay(15);
      } 
  }
}
void performMove(int initialPos,int finalPos){
  movePiece(initialPos,1);
  movePiece(finalPos,-1);
}
void loop(){
  while (true){
     if (Serial.available() > 0 && Serial.available()) {
     // read the incoming byte:
     incomingByte = Serial.read();
     if(incomingByte != 44 )
     {
      if(incomingByte == 46){
     num1 = pos[0].toInt();
     num2 = pos[1].toInt();
     num3 = pos[2].toInt();
     num4 = pos[3].toInt();
     // say what you got:
     //Serial.print("I got: "); // ASCII printable characters
     //Serial.println(incomingByte);
     //Serial.println(arr[0]);
     //Serial.println(arr[1]);
     if(num1==0){
      ind3++;
      pos[0] = " ";
      pos[1] = " ";
      pos[2] = " ";
      pos[3] = " ";
      ind1 = 0;
      continue;
     }
     arr[ind3/8][ind3%8][0] = num1;
     arr[ind3/8][ind3%8][1] = num2;
     arr[ind3/8][ind3%8][2] = num3;
     arr[ind3/8][ind3%8][3] = num4;
     movePiece(ind3,1);
     pos[0] = " ";
     pos[1] = " ";
     pos[2] = " ";
     pos[3] = " ";
     ind1 = 0;
     }
     else{
      pos[ind1] = pos[ind1] + (incomingByte - 48);
      }
    
     }
     else if(incomingByte == 44)
     {
      ind1++;
     }
    }
  } 
}


