#include <GL/glut.h>
#include <GL/gl.h>


int state = 0;
void flip(){
	if (state==0){
		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutSwapBuffers();
		state=1;
	}else{
		glClearColor(1,1,1,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutSwapBuffers();
		state=0;
	}
	glutTimerFunc(500,flip,1);
}

void display(){
	glutTimerFunc(500,flip,1);
}

void main(int argc, char**argv){
	glutInit(&argc, argv);
	glutInitWindowPosition(0,0);
	glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT));
	glutCreateWindow("flicker");
	glutDisplayFunc(display);
	glutMainLoop();
}




