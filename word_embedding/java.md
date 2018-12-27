1. What is JVM? **(1)**
    * The Java interpreter along with the runtime environment required to run the Java application is called as Java virtual machine(JVM).
    * A Java virtual machine (JVM) is an abstract computing machine that enables a computer to run a Java program.
    * The Java Virtual Machine (JVM) is the runtime engine of the Java Platform, which allows any program written in Java or other language compiled into Java bytecode to run on any computer that has a native JVM.
    * JVM Stands for Java Virtual Machine.  A JVM is a software-based machine that runs Java programs.
    * JVM Stands for Java Virtual Machine.

2.  What is a package? **(2)**
    * A package is a collection of related classes and interfaces providing access protection and namespace management.
    * A package is a namespace that organizes a set of related classes and interfaces.
    * A package is an organized and functionality based set of related interfaces and classes.
    * Java package is a group of similar types of classes, interfaces, and sub-packages.
    * A package, as the name suggests, is a pack (group) of classes, interfaces, and other packages.
    * package is a collection of classes and functions. 

3. What is an abstract class? **(3)**
    * An abstract class is a class designed with implementation gaps for subclasses. It is used for to fill in and is deliberately incomplete.
    * A class that is declared with abstract keyword, is known as an abstract class in java.
    * If a class contain any abstract method then the class is declared as an abstract class.  An abstract class is never instantiated.
    * An abstract class is a class that is declared abstract.  It may or may not include abstract methods.  Abstract classes cannot be instantiated, but they can be subclassed.
    * Abstract classes are classes that contain one or more abstract methods.

4. What is synchronization? **(4)**
    * Synchronization is the mechanism that ensures that only one thread is accessed the resources at a time.
    * Synchronization in Java is the capability to control the access of multiple threads to any shared resource.
    * Java is a multi-threaded language so, when two or more thread used a shared resources that lead to two kinds of errors. Thread interference and memory consistency error, to avoid this error you need to synchronized object. That the resource will be used by one thread at a time and the process by which synchronization is achieved is called synchronization.
    * Synchronization means executing threads in parallel.

5. What is JDBC? **(5)**
    * JDBC is a set of Java API for executing SQL statements.  This API consists of a set of classes and interfaces to enable programs to write pure Java Database applications.
    * Java Database Connectivity (JDBC) is an application programming interface (API) which allows the programmer to connect and interact with databases.
    * JDBC Stands for Java Database Connectivity.  JDBC is an API that allows Java applications to connect to and query a wide range of databases.
    * Java database connectivity (JDBC) is the Java Soft specification of a standard application programming interface (API) that allows Java programs to access database management systems.
    * Java JDBC is a Java API to connect and execute the query with the database.
    * Short for Java Database Connectivity, a Java API that enables Java programs to execute SQL statements.  This allows Java programs to interact with any SQL-compliant database.
    * Java Database Connectivity (JDBC) is an Application Programming Interface (API) used to connect Java application with Database.
    * JDBC stands for Java Database Connectivity.

6. What is an applet? **(6)**
    * An applet is a dynamic and interactive program that runs inside a web page displayed by a java capable browser.
    * An applet (little application) is a small software program that supports a larger application program.
    * A Java applet is a small application that is written in the Java programming language or another programming language that compiles to Java bytecode.
    * A Java applet is a small dynamic Java program that can be transferred via the Internet and run by a Java-compatible Web browser.
    * An applet is a small Internet-based program written in Java.

7. What is the Java API? **(7)**
    * The Java API is a large collection of ready-made software components that provide many useful capabilities, such as graphical user interface (GUI) widgets.
    * Java application programming interface (API) is a list of all classes that are part of the Java development kit (JDK).  It includes all Java packages, classes, and interfaces, along with their methods, fields, and constructors.
    * The Java API is the set of classes included with the Java Development Environment.
    * Java API is a collection of prewritten packages, classes, and interfaces with their respective methods, fields and constructors.

8. What is the difference between this() and super()? **(8)**
    * this() can be used to invoke a constructor of the same class whereas super() can be used to invoke a superclass constructor.
    * super() is used to call Base class’s constructor(ie, Parent’s class) while this() is used to call current class’s constructor.
    * Use of this() to call no argument constructor of the same class, while super() to call no argument or default constructor of parent class.

9. What is Constructor? **(9)**
    * A constructor is a special method whose task is to initialize the object of its class.  It is special because its name is the same as the class name.
    * A constructor in Java is a block of code similar to a method that's called when an instance of an object is created.
    * A constructor is a special member of a class which is used to initialize the state of an object.
    * Constructor in java is a special type of method that is used to initialize the object.

10. What is an Iterator? **(10)**
    * The Iterator interface is used to step through the elements of a Collection.
    * An iterator is an object that represents a stream of data Unlike a sequence, an iterator can (usually) only provide the next item.
    * An iterator is an object that enables a programmer to traverse a container.
    * Iterators are used in the Collection framework in Java for retrieving elements one by one from the collection..
    * Iterator allows us to traverse the collection, access the data element and remove the data elements of the collection.

11. What is JSP? **(11)**
    * JSP is a technology that returns dynamic content to the Web client using HTML, XML and JAVA elements JSP page looks like an HTML page but is a servlet.
    * JavaServer Pages (JSP) is a technology for developing Web pages that support dynamic content.
    * (JavaServer Page) An extension to the Java servlet technology that allows HTML to be combined with Java on the same page.
    * JavaServer Pages (JSP) is a complementary technology to Java Servlet which facilitates the mixing of dynamic and static web contents.

12. What is the purpose of apache tomcat? **(12)**
    * Apache server is a standalone server that is used to test servlets and create JSP pages.
    * Apache Tomcat is used to deploy your Java Servlets and JSPs.
    * Tomcat is an application server from the Apache Software Foundation that executes Java servlets and renders.  Web pages that include Java Server Page coding.
    * Apache Tomcat as a major pillar in developing the full stack of the web application.

13. Briefly explain daemon thread. **(13)**
    * The daemon thread is a low priority thread which runs in the background performs garbage collection operation for the java runtime system.
    * Daemon thread in java is a service provider thread that provides services to the user thread.  Its life depends on the mercy of user threads.
    * The thread that does not prevent the JVM from exiting when the program finishes but the thread is still running An example for a daemon thread is the garbage collection.
    * When we create a thread in Java, by default it’s a user thread and if it’s running JVM will not terminate the program.
    * A daemon thread is a thread that does not prevent the JVM from exiting when the program finishes but the thread is still running.

14. What is the 'final' keyword in Java used for? **(14)**
    * Final keyword is used with Class to make sure no other class can extend it, for example, String class is final and we can’t extend it.
    * We can use final keyword with methods to make sure child classes can’t override it.
    * Final keyword can be used with variables only once.  However the state of the variable can be changed, for example, we can assign a final variable to an object only once but the object variables can change later on.
    * Java interface variables are by default final and static.
    * When a variable is declared with final keyword, its value can't be modified, essentially, a constant.

15. What is a JAR file? **(15)**
    * JAR files are Java Archive files and it aggregates many files into one.
    * It holds Java classes in a library, JAR files are built in ZIP file format and have .jar file extension.
    * A JAR file (Java Archive) is a collection of Java code, a manifest file and other resources that together create a Java library, applet or JRE executable JAR file.
    * A JAR (Java Archive) is a package file format typically used to aggregate many Java class files and associated metadata and resources.

16. What is a JIT compiler? **(16)**
    * Just-In-Time (JIT) compiler is used to improve the performance.  JIT compiles parts of the bytecode that has similar functionality which in turn reduces the amount of time needed for compilation.
    * A JIT compiler runs after the program has started and compiles the code.
    * Just-In-Time (JIT) compiler is a program that turns Java bytecode into instructions that can be sent directly to the processor.
    * JIT compiler compiles the bytecodes of that method into native machine code.

17. What is Collections in java? **(17)**
    * The Collections API is a set of classes and interfaces that support operations on collections of objects.
    * Collections in java is a framework that provides an architecture to store and manipulate the group of objects.
    * The Java collections api provide java developers with a set of classes and interfaces that makes it easier to handle collections of objects.
    * The Java collections framework (JCF) is a set of classes and interfaces that implement commonly reusable collection data structures.

18. What is a servlet? **(18)**
    * A Java servlet is a Java program that extends the capabilities of a server.
    * A servlet is a small program that runs on a server.
    * A servlet is a Java program that runs on a Web server.
    * Java Servlets are programs that run on a Web or Application server and act as a middle layer between a request coming from a Web browser or other HTTP client and databases or applications on the HTTP server.

19. What is garbage collection? **(19)**
    * The process of removing unused objects from heap memory is known as Garbage collection
    * Java garbage collection is the process by which Java programs perform automatic memory management.
    * Removing unwanted objects or abandoned objects from the memory is called garbage collection.
    * Garbage collection is to free the memory By cleaning those objects that are no longer referenced by any of the programs.
    * The garbage collector is a program which gets rid of objects which are not being used by a Java application anymore.

20. What is JDK? **(20)**
    * JDK is an acronym for Java Development Kit.  It physically exists.  It contains JRE and development tools.
    * Java Development Kit (JDK) is a software development environment used for developing Java applications and applets.
    * A Java Development Kit (JDK) is a program development environment for writing Java applets and applications.
    * JDK (Java SE Development Kit) Includes a complete JRE (Java Runtime Environment) plus tools for developing, debugging, and monitoring Java applications.
    * JDK stands for Java Development Kit.

21. How does cookies work in Servlets? **(21)**
    * Cookies are text data sent by server to the client and it gets saved at the client's local machine.  Servlet API provides cookie support through javax.servlet.http.Cookie class that implements Serializable and Cloneable interfaces.
    * A cookie is a small piece of information that is persisted between the multiple client requests.
    * In cookie technique, we add cookie with response from the servlet.  So cookie is stored in the cache of the browser. After that if request is sent by the user, the cookie is added with the request by default.  Thus, we can recognize the user as the old user.
    * Cookies are small pieces of information that are sent in response from the web server to the client.

22. How is JSP better than servlet technology? **(22)**
    * JSP is a technology on the server’s side to make content generation simple.  They are document centric, whereas servlets are programs. A Java server page can contain fragments of a Java program, which execute and instantiates Java classes.
    * A servlet is a server-side program and written purely on Java.  JSP is an interface on top of Servlets.  In another way, we can say that JSPs are extension of servlets to minimize the effort of developers to write user interfaces using Java programming.
    * The advantage of JSP programming over servlets is that we can build custom tags which can directly call Java beans.
    * Servlets run faster compared to JSP. JSP can be compiled into Java Servlets.

23. Explain template method pattern in Java. **(23)**
    * Template pattern provides an outline of an algorithm and lets you configure or customise its steps.
    * Template method pattern is a behavioral design pattern that defines the program skeleton of an algorithm in an operation deferring some steps to subclasses.
    * Template Method is a behavioral design pattern that lets you define the skeleton of an algorithm and allow subclasses to redefine certain steps of the algorithm without changing its structure.
    * We can use Template method pattern to create an outline for an algorithm or a complex operation.

24. Why is Java platform independent? **(24)**
    * Platform independent practically means write once run anywhere.  Java is called so because of its byte code which can run on any system irrespective of its underlying operating system.
    * Platform independent means writing codes in one operating system and executing that code on another platform.
    * Write once and run anywhere.
    * Once a java code is compiled, compiled code (.class) can be executed on any os platform.
    * When you compile your java program, it generates a intermediate code called byte code which is called as platform independent.

25. Why there are no global variables in Java? **(25)**
    * Global variables are globally accessible.  Java does not support globally accessible variables as global variables break the referential transparency.
    * Global variables creates collisions in namespace.
    * Global Variables decreases the cohesion of a program.
    * In Java nothing is possible outside the class. So, Java Doesn’t have any term / concept of global variables.
