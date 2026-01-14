import React from 'react';
import { clsx } from 'clsx';

const Card = ({ children, className = '', ...props }) => {
  return (
    <div
      className={clsx(
        'bg-white overflow-hidden shadow rounded-lg',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

const CardHeader = ({ children, className = '', ...props }) => {
  return (
    <div
      className={clsx(
        'px-4 py-5 sm:px-6 border-b border-gray-200',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

const CardBody = ({ children, className = '', ...props }) => {
  return (
    <div
      className={clsx(
        'px-4 py-5 sm:p-6',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

const CardFooter = ({ children, className = '', ...props }) => {
  return (
    <div
      className={clsx(
        'px-4 py-4 sm:px-6 border-t border-gray-200',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;

export default Card;

